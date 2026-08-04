use std::env;

fn main() {
    let path = env::args().nth(1).expect("path");
    let target = u64::from_str_radix(
        env::args().nth(2).expect("target").trim_start_matches("0x"),
        16,
    )
    .expect("hex target");
    let executable = remu_loader::load_path(path).expect("load executable");
    let dynamic = executable
        .info
        .elf
        .expect("ELF")
        .dynamic
        .expect("dynamic metadata");
    for relocation in dynamic
        .relocations
        .iter()
        .chain(&dynamic.jump_relocations)
        .filter(|relocation| relocation.offset.get().abs_diff(target) <= 0x100)
    {
        let symbol = usize::try_from(relocation.symbol_index)
            .ok()
            .and_then(|index| dynamic.symbols.get(index));
        println!(
            "offset={:#x} type={} symbol={}:{} addend={:#x}",
            relocation.offset.get(),
            relocation.relocation_type,
            relocation.symbol_index,
            symbol.map_or("", |symbol| symbol.name.as_str()),
            relocation.addend,
        );
    }
    for (index, symbol) in dynamic.symbols.iter().enumerate().filter(|(_, symbol)| {
        symbol.value.get().abs_diff(target) <= 0x100
            || (symbol.value.get() <= target
                && symbol.value.get().saturating_add(symbol.size) > target)
    }) {
        println!(
            "symbol={index}:{} value={:#x} size={:#x} info={:#x} section={}",
            symbol.name,
            symbol.value.get(),
            symbol.size,
            symbol.info,
            symbol.section_index,
        );
    }
}
