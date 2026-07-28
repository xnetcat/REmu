use std::collections::BTreeSet;

use remu_core::GuestAddress;
use remu_loader::LoadedExecutable;
use remu_memory::{GuestImage, MappingError};

use crate::{LinkError, LinkedImage, runtime_address_at};

const PT_GNU_EH_FRAME: u32 = 0x6474_e550;

const DW_EH_PE_OMIT: u8 = 0xff;

const DW_EH_PE_INDIRECT: u8 = 0x80;

#[derive(Clone, Debug, Eq, PartialEq)]
pub(crate) struct GuestUnwindModule {
    pub(crate) name: String,
    pub(crate) base: u64,
    pub(crate) end: u64,
    pub(crate) eh_frame_header: u64,
    pub(crate) eh_frame: u64,
    pub(crate) eh_frame_size: u64,
}

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
type RegisterFrame = unsafe extern "C" fn(*const libc::c_void);

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
type RegisterEhFrameSection = unsafe extern "C" fn(*const libc::c_void, usize);

#[cfg(all(target_os = "macos", target_arch = "x86_64"))]
pub(crate) fn register_guest_unwind_frames(image: &LinkedImage) -> Result<usize, LinkError> {
    if image.unwind_fdes.is_empty() {
        return Ok(0);
    }
    let runtime = native_cxx_runtime().ok_or(LinkError::NativeCxxRuntimeUnavailable)?;
    // Modern Apple libunwind can ingest a complete .eh_frame section in one call. Silent Hill's
    // executable and bundled audio modules contain more than 450,000 FDEs; crossing the
    // Rust/libunwind boundary once per FDE adds a large serial delay before guest entry even
    // though libunwind immediately groups those records by section internally.
    //
    // LLVM added the explicit byte length to this entry point after the original one-argument
    // ABI. Supplying it is backwards-compatible on x86-64 (the older implementation ignores RSI)
    // and prevents a newer implementation from scanning section padding as another CIE.
    if let Some(register_section_address) =
        runtime.unwind_symbol(c"__unw_add_dynamic_eh_frame_section")
    {
        // SAFETY: dlsym returned Apple's libunwind section-registration entry point.
        let register_section: RegisterEhFrameSection = unsafe {
            std::mem::transmute(
                usize::try_from(register_section_address)
                    .expect("x86-64 unwind section registrar address fits usize"),
            )
        };
        for module in &image.unwind_modules {
            if module.eh_frame == 0 || module.eh_frame_size == 0 {
                continue;
            }
            // SAFETY: every module range was validated while linking and remains mapped for the
            // process lifetime. The complete section size is supplied for current libunwind.
            unsafe {
                register_section(
                    usize::try_from(module.eh_frame)
                        .expect("x86-64 guest .eh_frame address fits usize")
                        as *const libc::c_void,
                    usize::try_from(module.eh_frame_size)
                        .expect("x86-64 guest .eh_frame size fits usize"),
                );
            }
        }
        return Ok(image.unwind_fdes.len());
    }
    let register_address = runtime
        .unwind_symbol(c"__register_frame")
        .ok_or(LinkError::MissingNativeUnwindSymbol("__register_frame"))?;
    // SAFETY: dlsym returned the address of libunwind's documented single-pointer function.
    let register: RegisterFrame = unsafe {
        std::mem::transmute(
            usize::try_from(register_address).expect("x86-64 unwind address fits usize"),
        )
    };
    for fde in &image.unwind_fdes {
        // SAFETY: Every address was decoded and bounds-checked from the mapped guest .eh_frame.
        unsafe {
            register(
                usize::try_from(fde.get()).expect("x86-64 guest address fits usize")
                    as *const libc::c_void,
            );
        }
    }
    Ok(image.unwind_fdes.len())
}

pub(crate) fn collect_unwind_fdes_at(
    memory: &GuestImage,
    image_base: GuestAddress,
    executable: &LoadedExecutable,
) -> Result<Vec<GuestAddress>, LinkError> {
    let Some(header) = executable
        .info
        .elf
        .as_ref()
        .and_then(|elf| {
            elf.program_headers
                .iter()
                .find(|header| header.program_type == PT_GNU_EH_FRAME)
        })
        .filter(|header| header.file_size >= 4)
    else {
        return Ok(Vec::new());
    };
    let header_size = usize::try_from(header.file_size)
        .map_err(|_| LinkError::MalformedEhFrame(".eh_frame_hdr is too large"))?;
    let header_address =
        runtime_address_at(memory, image_base, header.virtual_address, header_size)?;
    let mut header_bytes = vec![0_u8; header_size];
    memory.read(header_address, &mut header_bytes)?;
    if header_bytes[0] != 1 {
        return Err(LinkError::MalformedEhFrame(
            "unsupported .eh_frame_hdr version",
        ));
    }
    let mut cursor = 4;
    let frame_address = decode_eh_pointer(
        memory,
        &header_bytes,
        &mut cursor,
        header_bytes[1],
        header_address,
    )?;
    if !memory.contains(GuestAddress::new(frame_address), 4) {
        return Err(LinkError::MalformedEhFrame(
            ".eh_frame pointer is outside the process image",
        ));
    }
    let expected_fdes = if header_bytes[2] == DW_EH_PE_OMIT {
        0
    } else {
        usize::try_from(decode_eh_pointer(
            memory,
            &header_bytes,
            &mut cursor,
            header_bytes[2],
            header_address,
        )?)
        .map_err(|_| LinkError::MalformedEhFrame("FDE count is too large"))?
    };
    let table_encoding = header_bytes[3];
    if expected_fdes != 0 && table_encoding == DW_EH_PE_OMIT {
        return Err(LinkError::MalformedEhFrame(
            "nonempty .eh_frame_hdr omits its search table",
        ));
    }
    let mut fdes = Vec::with_capacity(expected_fdes);
    let mut unique_fdes = BTreeSet::new();
    for _ in 0..expected_fdes {
        let _initial_location = decode_eh_pointer(
            memory,
            &header_bytes,
            &mut cursor,
            table_encoding,
            header_address,
        )?;
        let fde = decode_eh_pointer(
            memory,
            &header_bytes,
            &mut cursor,
            table_encoding,
            header_address,
        )?;
        validate_fde(memory, fde)?;
        if unique_fdes.insert(fde) {
            fdes.push(GuestAddress::new(fde));
        }
    }
    if fdes.len() != expected_fdes {
        return Err(LinkError::MalformedEhFrame(
            ".eh_frame_hdr contains duplicate FDE pointers",
        ));
    }
    Ok(fdes)
}

pub(crate) fn guest_unwind_module(
    memory: &GuestImage,
    name: &str,
    image_base: GuestAddress,
    executable: &LoadedExecutable,
) -> Result<GuestUnwindModule, LinkError> {
    let mut base = u64::MAX;
    let mut end = 0_u64;
    for segment in &executable.load_segments {
        if segment.memory_size == 0 {
            continue;
        }
        let start = runtime_address_at(
            memory,
            image_base,
            segment.virtual_address,
            usize::try_from(segment.memory_size).map_err(|_| LinkError::AddressOverflow)?,
        )?
        .get();
        base = base.min(start);
        end = end.max(
            start
                .checked_add(segment.memory_size)
                .ok_or(LinkError::AddressOverflow)?,
        );
    }
    if base == u64::MAX || end <= base {
        return Err(LinkError::Mapping(MappingError::NoLoadSegments));
    }

    let mut module = GuestUnwindModule {
        name: name.to_owned(),
        base,
        end,
        eh_frame_header: 0,
        eh_frame: 0,
        eh_frame_size: 0,
    };
    let Some(header) = executable.info.elf.as_ref().and_then(|elf| {
        elf.program_headers
            .iter()
            .find(|header| header.program_type == PT_GNU_EH_FRAME && header.file_size >= 8)
    }) else {
        return Ok(module);
    };
    let header_size = usize::try_from(header.file_size)
        .map_err(|_| LinkError::MalformedEhFrame(".eh_frame_hdr is too large"))?;
    let header_address =
        runtime_address_at(memory, image_base, header.virtual_address, header_size)?;
    let mut header_bytes = vec![0_u8; header_size];
    memory.read(header_address, &mut header_bytes)?;
    if header_bytes[0] != 1 {
        return Err(LinkError::MalformedEhFrame(
            "unsupported .eh_frame_hdr version",
        ));
    }
    let mut cursor = 4;
    let frame_address = decode_eh_pointer(
        memory,
        &header_bytes,
        &mut cursor,
        header_bytes[1],
        header_address,
    )?;
    if frame_address < base || frame_address >= end {
        return Err(LinkError::MalformedEhFrame(
            ".eh_frame pointer is outside its module",
        ));
    }
    module.eh_frame_header = header_address.get();
    module.eh_frame = frame_address;
    module.eh_frame_size = if header_address.get() > frame_address {
        header_address.get() - frame_address
    } else {
        end - frame_address
    };
    Ok(module)
}

fn decode_eh_pointer(
    memory: &GuestImage,
    bytes: &[u8],
    cursor: &mut usize,
    encoding: u8,
    data_relative_base: GuestAddress,
) -> Result<u64, LinkError> {
    if encoding == DW_EH_PE_OMIT {
        return Err(LinkError::UnsupportedEhFrameEncoding(encoding));
    }
    let field_address = data_relative_base
        .get()
        .checked_add(u64::try_from(*cursor).map_err(|_| LinkError::AddressOverflow)?)
        .ok_or(LinkError::AddressOverflow)?;
    let (raw, width, signed) = match encoding & 0x0f {
        0x00 | 0x04 => (read_eh_unsigned(bytes, *cursor, 8)?, 8, false),
        0x02 => (read_eh_unsigned(bytes, *cursor, 2)?, 2, false),
        0x03 => (read_eh_unsigned(bytes, *cursor, 4)?, 4, false),
        0x0a => (read_eh_unsigned(bytes, *cursor, 2)?, 2, true),
        0x0b => (read_eh_unsigned(bytes, *cursor, 4)?, 4, true),
        0x0c => (read_eh_unsigned(bytes, *cursor, 8)?, 8, true),
        _ => return Err(LinkError::UnsupportedEhFrameEncoding(encoding)),
    };
    *cursor = cursor
        .checked_add(width)
        .ok_or(LinkError::AddressOverflow)?;
    let value = if signed {
        sign_extend_eh_value(raw, width)
    } else {
        i128::from(raw)
    };
    let base = match encoding & 0x70 {
        0x00 => 0_i128,
        0x10 => i128::from(field_address),
        0x30 => i128::from(data_relative_base.get()),
        _ => return Err(LinkError::UnsupportedEhFrameEncoding(encoding)),
    };
    let resolved = base.checked_add(value).ok_or(LinkError::MalformedEhFrame(
        "encoded unwind pointer overflows the guest address space",
    ))?;
    let resolved = u64::try_from(resolved).map_err(|_| {
        LinkError::MalformedEhFrame("encoded unwind pointer overflows the guest address space")
    })?;
    if encoding & DW_EH_PE_INDIRECT == 0 {
        return Ok(resolved);
    }
    let mut indirect = [0_u8; 8];
    memory.read(GuestAddress::new(resolved), &mut indirect)?;
    Ok(u64::from_le_bytes(indirect))
}

fn read_eh_unsigned(bytes: &[u8], offset: usize, width: usize) -> Result<u64, LinkError> {
    let end = offset
        .checked_add(width)
        .ok_or(LinkError::AddressOverflow)?;
    let source = bytes.get(offset..end).ok_or(LinkError::MalformedEhFrame(
        "encoded unwind pointer exceeds .eh_frame_hdr",
    ))?;
    let mut value = [0_u8; 8];
    value[..width].copy_from_slice(source);
    Ok(u64::from_le_bytes(value))
}

fn sign_extend_eh_value(value: u64, width: usize) -> i128 {
    let shift = 64 - width * 8;
    i128::from((value << shift).cast_signed() >> shift)
}

fn validate_fde(memory: &GuestImage, address: u64) -> Result<(), LinkError> {
    let mut short_header = [0_u8; 12];
    memory.read(GuestAddress::new(address), &mut short_header)?;
    let short_length = u32::from_le_bytes(short_header[..4].try_into().expect("FDE length"));
    if short_length == 0 {
        return Err(LinkError::MalformedEhFrame("FDE has a zero length"));
    }
    let (length_field_size, entry_length, cie_pointer_size, cie_pointer) =
        if short_length == u32::MAX {
            (
                12_u64,
                u64::from_le_bytes(short_header[4..12].try_into().expect("extended FDE length")),
                8_usize,
                None,
            )
        } else {
            (
                4_u64,
                u64::from(short_length),
                4_usize,
                Some(u64::from(u32::from_le_bytes(
                    short_header[4..8].try_into().expect("FDE CIE pointer"),
                ))),
            )
        };
    if entry_length < cie_pointer_size as u64 {
        return Err(LinkError::MalformedEhFrame(
            "FDE is shorter than its CIE pointer",
        ));
    }
    let entry_size = length_field_size
        .checked_add(entry_length)
        .ok_or(LinkError::MalformedEhFrame("FDE size overflow"))?;
    let entry_size =
        usize::try_from(entry_size).map_err(|_| LinkError::MalformedEhFrame("FDE is too large"))?;
    if !memory.contains(GuestAddress::new(address), entry_size) {
        return Err(LinkError::MalformedEhFrame(
            "FDE extends outside the process image",
        ));
    }
    let cie_pointer = if let Some(pointer) = cie_pointer {
        pointer
    } else {
        let mut pointer = [0_u8; 8];
        let pointer_address = address
            .checked_add(length_field_size)
            .ok_or(LinkError::MalformedEhFrame("FDE address overflow"))?;
        memory.read(GuestAddress::new(pointer_address), &mut pointer)?;
        u64::from_le_bytes(pointer)
    };
    if cie_pointer == 0 {
        return Err(LinkError::MalformedEhFrame(
            ".eh_frame_hdr table points to a CIE instead of an FDE",
        ));
    }
    Ok(())
}
