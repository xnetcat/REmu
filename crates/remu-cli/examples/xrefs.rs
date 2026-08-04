use iced_x86::{Decoder, DecoderOptions, Formatter, IntelFormatter};
use std::env;

fn main() {
    let path = env::args().nth(1).expect("path");
    let target = u64::from_str_radix(
        env::args().nth(2).expect("target").trim_start_matches("0x"),
        16,
    )
    .expect("hex target");
    let executable = remu_loader::load_path(path).expect("load executable");
    let base = remu_memory::PS5_MAIN_IMAGE_BASE.get();
    let mut formatter = IntelFormatter::new();
    let mut formatted = String::new();
    for segment in executable
        .load_segments
        .iter()
        .filter(|segment| segment.flags & 1 != 0)
    {
        let mapped = base + segment.virtual_address.get();
        let mut decoder = Decoder::with_ip(64, &segment.data, mapped, DecoderOptions::NONE);
        while decoder.can_decode() {
            let instruction = decoder.decode();
            if instruction.is_ip_rel_memory_operand()
                && instruction.ip_rel_memory_address().abs_diff(target) <= 0x100
            {
                formatted.clear();
                formatter.format(&instruction, &mut formatted);
                println!(
                    "{:#018x} -> {:#018x} {formatted}",
                    instruction.ip(),
                    instruction.ip_rel_memory_address(),
                );
            }
        }
    }
}
