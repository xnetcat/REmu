use super::*;

struct CommandFixture {
    descriptor: Box<[u64; 7]>,
    words: Box<[u32; 128]>,
}

impl CommandFixture {
    fn new(reserved: u32) -> Self {
        let descriptor = Box::new([0_u64; 7]);
        let words = Box::new([0_u32; 128]);
        let mut fixture = Self { descriptor, words };
        let start = fixture.words.as_mut_ptr().addr() as u64;
        fixture.descriptor[0] = start;
        fixture.descriptor[1] = start + fixture.words.len() as u64 * 4;
        fixture.descriptor[2] = start;
        fixture.descriptor[3] = start + fixture.words.len() as u64 * 4;
        fixture.descriptor[6] = u64::from(reserved);
        fixture
    }

    fn address(&mut self) -> u64 {
        self.descriptor.as_mut_ptr().addr() as u64
    }

    fn cursor(&self) -> u64 {
        self.descriptor[2]
    }

    fn start(&self) -> u64 {
        self.words.as_ptr().addr() as u64
    }
}

fn args(values: &[u64]) -> [u64; 6] {
    let mut result = [0_u64; 6];
    result[..values.len()].copy_from_slice(values);
    result
}

fn call(nid: &str, arguments: [u64; 6], stack: [u64; 6]) -> u64 {
    dispatch_agc_command_abi(nid, arguments, stack).expect("known test NID")
}

#[test]
fn returns_exact_fixed_and_variable_packet_sizes_in_bytes() {
    assert_eq!(call("Abendgtz+3o", [0; 6], [0; 6]), 20);
    assert_eq!(call("-vnlTPPXPrw", [0; 6], [0; 6]), 32);
    assert_eq!(call("6ee9Hd3EWXQ", [0; 6], [0; 6]), 24);
    assert_eq!(call("t7PlZ9nt5Lc", args(&[13]), [0; 6]), 52);
    assert_eq!(call("bxGoVxpdSPQ", args(&[7]), [0; 6]), 36);
    assert_eq!(call("p9tI+yTvx68", args(&[9]), [0; 6]), 52);
    assert_eq!(
        dispatch_agc_command_abi("not-an-agc-nid", [0; 6], [0; 6]),
        None
    );
}

#[test]
fn records_a_branch_target_that_stitches_into_the_next_dcb() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();
    let compare = 0_u64;
    let compare_address = std::ptr::from_ref(&compare).addr() as u64;
    let target = Box::new([pm4(3, IT_SET_SH_REG, 0), 0x0c, 0x1234_5678, 0]);
    let target_address = target.as_ptr().addr() as u64;
    let next_dcb = target_address + 12;

    let command = call(
        "w1KFAHVqpaU",
        args(&[buffer, 1, 0, compare_address, 0, 0]),
        args(&[0, target_address, 3, 0, 0, 0]),
    );

    assert_eq!(command, fixture.start());
    assert_eq!(fixture.words[0], pm4(14, IT_INDIRECT_BUFFER, 0));
    assert_eq!(take_graphics_continuations_before(next_dcb), vec![command]);
    assert!(take_graphics_continuations_before(next_dcb).is_empty());
    assert!(graphics_continuation_was_consumed(command));
}

#[test]
fn explicitly_submitted_branch_target_is_not_replayed_before_the_next_dcb() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();
    let compare = 0_u64;
    let compare_address = std::ptr::from_ref(&compare).addr() as u64;
    let target = Box::new([pm4(3, IT_SET_SH_REG, 0), 0x0c, 0x1234_5678]);
    let target_address = target.as_ptr().addr() as u64;
    let next_dcb = target_address + 12;

    let command = call(
        "w1KFAHVqpaU",
        args(&[buffer, 1, 0, compare_address, 0, 0]),
        args(&[0, target_address, 3, 0, 0, 0]),
    );
    track_graphics_submission(target_address, 3);

    assert!(take_graphics_continuations_before(next_dcb).is_empty());
    assert!(graphics_continuation_was_consumed(command));
}

#[test]
fn directly_executed_branch_is_not_recovered_again_before_the_next_dcb() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();
    let compare = 0_u64;
    let compare_address = std::ptr::from_ref(&compare).addr() as u64;
    let target = Box::new([pm4(3, IT_SET_SH_REG, 0), 0x0c, 0x1234_5678]);
    let target_address = target.as_ptr().addr() as u64;
    let next_dcb = target_address + 12;

    let command = call(
        "w1KFAHVqpaU",
        args(&[buffer, 1, 0, compare_address, 0, 0]),
        args(&[0, target_address, 3, 0, 0, 0]),
    );
    track_graphics_allocation(target_address, 3);
    retire_pending_graphics_continuation(command);

    assert!(take_graphics_continuations_before(next_dcb).is_empty());
    assert!(!graphics_continuation_was_consumed(command));
    assert_eq!(take_recorded_graphics_prefix(next_dcb), None);
}

#[test]
fn recovers_only_a_complete_recorded_pm4_prefix() {
    let commands = Box::new([
        pm4(3, IT_SET_SH_REG, 0),
        0x0c,
        0x1111_1111,
        pm4(3, IT_SET_SH_REG, 0),
        0x0d,
        0x2222_2222,
        pm4(3, IT_SET_SH_REG, 0),
        0x0e,
        0x3333_3333,
    ]);
    let start = commands.as_ptr().addr() as u64;
    track_graphics_allocation(start, 3);
    track_graphics_allocation(start + 12, 3);

    assert_eq!(take_recorded_graphics_prefix(start + 24), Some((start, 6)));
    assert_eq!(take_recorded_graphics_prefix(start + 24), None);
}

#[test]
fn a_recorded_anchor_recovers_neighboring_guest_authored_pm4() {
    let commands = Box::new([
        pm4(3, IT_SET_SH_REG, 0),
        0x0c,
        0x1111_1111,
        pm4(3, IT_SET_SH_REG, 0),
        0x0d,
        0x2222_2222,
        pm4(3, IT_SET_SH_REG, 0),
        0x0e,
        0x3333_3333,
    ]);
    let start = commands.as_ptr().addr() as u64;
    track_graphics_allocation(start, 3);
    track_graphics_allocation(start + 24, 3);

    assert_eq!(take_recorded_graphics_prefix(start + 36), Some((start, 9)));
}

#[test]
fn consumed_continuation_target_cannot_be_replayed_as_a_prefix() {
    let commands = Box::new([
        pm4(3, IT_SET_SH_REG, 0),
        0x0c,
        0x1111_1111,
        pm4(3, IT_SET_SH_REG, 0),
        0x0d,
        0x2222_2222,
    ]);
    let start = commands.as_ptr().addr() as u64;
    track_graphics_allocation(start, 3);
    track_graphics_allocation(start + 12, 3);

    consume_recorded_graphics_range(start, 6);

    assert_eq!(take_recorded_graphics_prefix(start + 24), None);
}

#[test]
fn submitted_commands_are_not_replayed_before_a_later_tail() {
    let commands = Box::new([
        pm4(3, IT_SET_SH_REG, 0),
        0x0c,
        0x1111_1111,
        pm4(3, IT_SET_SH_REG, 0),
        0x0d,
        0x2222_2222,
    ]);
    let start = commands.as_ptr().addr() as u64;
    track_graphics_allocation(start, 3);
    track_graphics_allocation(start + 12, 3);

    track_graphics_submission(start, 3);

    assert_eq!(
        take_recorded_graphics_prefix(start + 24),
        Some((start + 12, 3))
    );
}

#[test]
fn async_allocations_are_not_recorded_as_a_graphics_prefix() {
    let mut fixture = CommandFixture::new(0);
    let descriptor = fixture.address();
    let graphics = call("TRO721eVt4g", args(&[descriptor, 0, 0]), [0; 6]);
    assert_eq!(graphics, fixture.start());
    track_graphics_submission(graphics, 2);

    let compute = call("JrtiDtKeS38", args(&[descriptor, 0]), [0; 6]);

    assert_eq!(compute, graphics + 8);
    assert_eq!(take_recorded_graphics_prefix(compute + 8), None);
}

#[test]
fn a_higher_submission_does_not_discard_a_wrapped_lower_prefix() {
    let commands = Box::new([
        pm4(3, IT_SET_SH_REG, 0),
        0x0c,
        0x1111_1111,
        0,
        0,
        0,
        pm4(3, IT_SET_SH_REG, 0),
        0x0d,
        0x2222_2222,
        0,
        0,
        0,
    ]);
    let start = commands.as_ptr().addr() as u64;
    let higher_submission = start + 24;
    track_graphics_allocation(start, 3);

    track_graphics_submission(higher_submission, 3);

    assert_eq!(take_recorded_graphics_prefix(start + 12), Some((start, 3)));
}

#[test]
fn builds_native_register_and_draw_packets() {
    let mut fixture = CommandFixture::new(2);
    let buffer = fixture.address();
    let register = u64::from(0xdead_beef_u32) << 32 | 0x191;

    assert_eq!(
        call("LHFXRrlTPD8", args(&[buffer, register]), [0; 6]),
        fixture.start()
    );
    assert_eq!(
        &fixture.words[..3],
        &[pm4(3, IT_SET_CONTEXT_REG, 0), 0x191, 0xdead_beef]
    );

    let registers = [0x123_u32, 0x456, 0x124, 0x789];
    let register_address = registers.as_ptr().addr() as u64;
    assert_eq!(
        call("-HOOCn0JY48", args(&[buffer, register_address, 2]), [0; 6]),
        fixture.start() + 12
    );
    assert_eq!(
        &fixture.words[3..8],
        &[
            pm4(5, IT_SET_SH_REG_INDIRECT, 0),
            low_u32(register_address) & !3,
            low_u32(register_address >> 32),
            0x8000_0000,
            2,
        ]
    );

    assert_eq!(
        call("Yw0jKSqop+E", args(&[buffer, 6, 0]), [0; 6]),
        fixture.start() + 32
    );
    assert_eq!(
        &fixture.words[8..11],
        &[pm4(3, IT_DRAW_INDEX_AUTO, 0), 6, 2]
    );
    assert_eq!(fixture.cursor(), fixture.start() + 44);
}

#[test]
fn builds_native_type_two_packet() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();

    assert_eq!(
        call("qj7QZpgr9Uw", args(&[buffer]), [0; 6]),
        fixture.start()
    );
    assert_eq!(fixture.words[0], 0x8000_0000);
    assert_eq!(fixture.cursor(), fixture.start() + 4);
}

#[test]
fn patches_single_and_range_packet_predication_bits() {
    let mut packet = [pm4(3, IT_SET_SH_REG, 0), 0x123, 0x456];
    let address = packet.as_mut_ptr().addr() as u64;

    assert_eq!(call("w6Dj1VJt5qY", args(&[address, 0x101]), [0; 6]), 0);
    assert_eq!(packet[0], pm4(3, IT_SET_SH_REG, 0) | 1);
    assert_eq!(call("w6Dj1VJt5qY", args(&[address, 2]), [0; 6]), 0);
    assert_eq!(packet[0], pm4(3, IT_SET_SH_REG, 0));
    assert_eq!(
        call("w6Dj1VJt5qY", args(&[0, 1]), [0; 6]),
        GRAPHICS5_ERROR_INVALID_PACKET
    );

    let mut range = [
        pm4(2, IT_CLEAR_STATE, 0),
        0x0f,
        pm4(3, IT_SET_SH_REG, 0),
        0x123,
        0x456,
    ];
    let start = range.as_mut_ptr().addr() as u64;
    let end = start + u64::try_from(range.len() * 4).expect("range byte count fits u64");
    assert_eq!(call("n8vgpaQg6dA", args(&[start, end, 1]), [0; 6]), 0);
    assert_eq!(range[0], pm4(2, IT_CLEAR_STATE, 0) | 1);
    assert_eq!(range[2], pm4(3, IT_SET_SH_REG, 0) | 1);
    assert_eq!(&range[1..2], &[0x0f]);
    assert_eq!(&range[3..], &[0x123, 0x456]);
}

#[test]
fn reserves_a_register_range_payload_when_values_are_null() {
    let mut fixture = CommandFixture::new(0);
    fixture.words[2..6].copy_from_slice(&[0x1111, 0x2222, 0x3333, 0x4444]);
    let buffer = fixture.address();

    assert_eq!(
        call("n2fD4A+pb+g", args(&[buffer, 0x8c, 0, 4]), [0; 6]),
        fixture.start()
    );
    assert_eq!(fixture.words[0], pm4(6, IT_SET_SH_REG, 0));
    assert_eq!(fixture.words[1], 0x8c);
    assert_eq!(&fixture.words[2..6], &[0x1111, 0x2222, 0x3333, 0x4444]);
    assert_eq!(fixture.cursor(), fixture.start() + 24);
}

#[test]
fn preserves_indirect_register_packet_fields_while_patching() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();
    let registers = [0_u64; 4];
    let first = registers.as_ptr().addr() as u64;
    let command = call("ZvwO9euwYzc", args(&[buffer, first, 7]), [0; 6]);
    fixture.words[4] |= 0xc000_0000;
    let second = first + 0x20;

    assert_eq!(call("vcmNN+AAXnY", args(&[command, second]), [0; 6]), 0);
    assert_eq!(call("d-6uF9sZDIU", args(&[command, 9]), [0; 6]), 0);
    assert_eq!(fixture.words[1], low_u32(second) & !3);
    assert_eq!(fixture.words[2], low_u32(second >> 32));
    assert_eq!(fixture.words[4], 0xc000_0010);
    assert_eq!(
        call("Qrj4c+61z4A", args(&[command, second]), [0; 6]),
        GRAPHICS5_ERROR_INVALID_PACKET
    );
}

#[test]
fn builds_and_patches_native_dma_data() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();
    let command = call(
        "WmAc2MEj6Io",
        args(&[buffer, 1, 2, 3, 0x1122_3344_5566_7788, 1]),
        args(&[2, 0x8877_6655_4433_2211, 0x3456, 1, 1, 1]),
    );
    assert_eq!(command, fixture.start());
    assert_eq!(fixture.words[0], pm4(7, IT_DMA_DATA, 0));
    assert_eq!(
        &fixture.words[2..6],
        &[0x4433_2211, 0x8877_6655, 0x5566_7788, 0x1122_3344]
    );
    assert_eq!(
        call(
            "IxYiarKlXxM",
            args(&[command, 0xaabb_ccdd_eeff_0010]),
            [0; 6]
        ),
        0
    );
    assert_eq!(
        call(
            "cdDRpqcFGbU",
            args(&[command, 0x1234_5678_9abc_def0]),
            [0; 6]
        ),
        0
    );
    assert_eq!(
        &fixture.words[2..6],
        &[0x9abc_def0, 0x1234_5678, 0xeeff_0010, 0xaabb_ccdd]
    );
}

#[test]
fn builds_async_dma_data_with_the_queue_implied_engine() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();
    let destination = 0x1234_5678_9abc_def0;
    let command = call(
        "-RnpfpxIhec",
        args(&[buffer, 3, 3, destination, 2, 3]),
        args(&[1, 4, 0, 0]),
    );

    assert_eq!(command, fixture.start());
    assert_eq!(fixture.words[0], pm4(7, IT_DMA_DATA, 0));
    assert_eq!(fixture.words[1] & 1, 1);
    assert_eq!((fixture.words[1] >> 20) & 3, 3);
    assert_eq!((fixture.words[1] >> 29) & 3, 2);
    assert_eq!(&fixture.words[2..4], &[1, 0]);
    assert_eq!(
        &fixture.words[4..6],
        &[low_u32(destination), low_u32(destination >> 32)]
    );
    assert_eq!(fixture.words[6] & 0x03ff_ffff, 4);
}

#[test]
fn builds_and_patches_64_bit_wait_packet() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();
    let wait_address = 0x1234_5678_9abc_def0;
    let command = call(
        "VmW0Tdpy420",
        args(&[buffer, 1, 2, 5, 3, wait_address]),
        args(&[0x1111_2222_3333_4444, 0xffff_ffff_0000_0000, 0x12345]),
    );
    assert_eq!(command, fixture.start());
    assert_eq!(fixture.words[0], pm4(9, IT_NOP, R_WAIT_MEM_64));
    assert_eq!(fixture.words[8], 0x1234);

    assert_eq!(call("n485EBnIWmk", args(&[command, 7]), [0; 6]), 0);
    assert_eq!(
        call(
            "7nOoijNPvEU",
            args(&[command, 0xaaaa_bbbb_cccc_dddd]),
            [0; 6]
        ),
        0
    );
    assert_eq!(
        call(
            "hXAnLgDHCoI",
            args(&[command, 0x0123_4567_89ab_cdef]),
            [0; 6]
        ),
        0
    );
    assert_eq!(fixture.words[7] & 7, 7);
    assert_eq!(
        &fixture.words[3..7],
        &[0x89ab_cdef, 0x0123_4567, 0xcccc_dddd, 0xaaaa_bbbb]
    );
    assert_eq!(
        call("3KDcnM3lrcU", args(&[command, wait_address | 4]), [0; 6]),
        GRAPHICS5_ERROR_INVALID_PACKET
    );
}

#[test]
fn release_mem_and_patchers_share_the_native_layout() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();
    let command = call(
        "wr23dPKyWc0",
        args(&[buffer, 0x2f, 0x100, 1, 2, 0x70_0000_0001]),
        args(&[1, 0x1122_3344_5566_7788, 0, 0, 2, 0x123]),
    );
    assert_eq!(command, fixture.start());
    assert_eq!(fixture.words[0], pm4(8, IT_NOP, R_RELEASE_MEM));
    assert_eq!(fixture.words[1], 0x22f);
    assert_eq!(fixture.words[2], 0x0201_0100);
    assert_eq!(&fixture.words[3..5], &[1, 0x70]);

    assert_eq!(call("J8YCgfKAMQs", args(&[command, 0x456]), [0; 6]), 0);
    assert_eq!(call("T9fjQIINoeE", args(&[command, 3]), [0; 6]), 0);
    assert_eq!(
        call("0fWWK5uG9rQ", args(&[command, 0x70_0000_0001]), [0; 6]),
        0
    );
    assert_eq!(
        call("MlEw1feXcjg", args(&[command, 2, 1, 0x00ab_cdef]), [0; 6]),
        0
    );
    assert_eq!(fixture.words[2] & 0xffff, 0x456);
    assert_eq!((fixture.words[2] >> 16) & 0xff, 3);
    assert_eq!(&fixture.words[3..7], &[1, 0x70, 0x00ab_cdef, 0]);
}

#[test]
fn command_buffer_reservation_and_alignment_are_enforced() {
    let mut fixture = CommandFixture::new(127);
    let buffer = fixture.address();
    let start = fixture.start();
    assert_eq!(call("Yw0jKSqop+E", args(&[buffer, 3, 0]), [0; 6]), 0);
    assert_eq!(fixture.cursor(), start);

    fixture.descriptor[6] = 0;
    fixture.descriptor[3] = start + 8;
    assert_eq!(call("Yw0jKSqop+E", args(&[buffer, 3, 0]), [0; 6]), 0);
    assert_eq!(fixture.cursor(), start);

    fixture.descriptor[3] = start + 12;
    assert_eq!(call("Yw0jKSqop+E", args(&[buffer, 3, 0]), [0; 6]), start);
    assert_eq!(fixture.cursor(), start + 12);
    assert_eq!(call("Yw0jKSqop+E", args(&[buffer + 4, 3, 0]), [0; 6]), 0);
}

#[test]
fn invokes_the_gen5_growth_callback_before_allocating() {
    unsafe extern "C" fn grow(command_buffer: u64, requested: u32, user_data: u64) -> bool {
        if requested != 7 {
            return false;
        }
        write_u64(
            command_buffer + COMMAND_BUFFER_CURSOR_UP_OFFSET,
            read_u64(user_data),
        );
        write_u64(
            command_buffer + COMMAND_BUFFER_CURSOR_DOWN_OFFSET,
            read_u64(user_data + 8),
        );
        true
    }

    let mut fixture = CommandFixture::new(2);
    let buffer = fixture.address();
    let start = fixture.start();
    let extent = Box::new([start, start + 28]);
    fixture.descriptor[3] = start + 4;
    fixture.descriptor[4] = grow as *const () as usize as u64;
    fixture.descriptor[5] = extent.as_ptr().addr() as u64;

    assert_eq!(
        call("k3GhuSNmBLU", args(&[buffer, 2, 3, 4, 0]), [0; 6]),
        start
    );
    assert_eq!(fixture.cursor(), start + 20);
    assert_eq!(fixture.words[0], pm4(5, IT_DISPATCH_DIRECT, 0));
}

#[test]
fn four_argument_index_buffer_builder_remains_stateful() {
    assert_eq!(
        dispatch_agc_command_abi("l4fM9K-Lyks", [0; 6], [0; 6]),
        None
    );
    assert_eq!(
        dispatch_agc_command_abi("j4emHHndCPY", [0; 6], [0; 6]),
        None
    );
}

#[test]
fn writes_inline_data_and_marker_payloads() {
    let mut fixture = CommandFixture::new(0);
    let buffer = fixture.address();
    let data = [0x1122_3344_u32, 0xaabb_ccdd];
    let command = call(
        "i1jyy49AjXU",
        args(&[buffer, 2, 1, 0x4000, data.as_ptr().addr() as u64, 2]),
        args(&[1, 1]),
    );
    assert_eq!(command, fixture.start());
    assert_eq!(fixture.words[0], pm4(6, IT_WRITE_DATA, 0));
    assert_eq!(&fixture.words[4..6], &data);

    let marker = b"frame-start\0";
    let marker_command = call(
        "+kSrjIVxKFE",
        args(&[buffer, marker.as_ptr().addr() as u64]),
        [0; 6],
    );
    assert_eq!(marker_command, fixture.start() + 24);
    let payload =
        unsafe { std::slice::from_raw_parts((marker_command + 4) as *const u8, marker.len()) };
    assert_eq!(payload, marker);
}
