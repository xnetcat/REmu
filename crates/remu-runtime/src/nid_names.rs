//! Symbol names recovered for imported NIDs.
//!
//! A PS5 import carries no name, only an 11-character NID: the first eight bytes of
//! `SHA-1(symbol || salt)`, little-endian, in Sony's base64 alphabet.  The hash is one-way, so a
//! name can only be recovered by hashing candidate symbols and keeping the ones that match.
//! `scripts/generate-nid-names.py` does exactly that across a directory of PS5 modules and writes
//! the matches to the table this module reads.
//!
//! The table is consulted only after the hand-maintained one in [`crate::known_nid_name`] misses,
//! and it feeds diagnostics alone -- an entry here never makes an import dispatch anywhere.  An
//! unimplemented import stays unimplemented; it just gets reported by name rather than as an
//! opaque NID, which is the difference between a log line someone can act on and one they cannot.
//!
//! It ships as a binary blob rather than Rust source because the recovered set runs to tens of
//! thousands of entries, and that many string literals costs far more build time than it is worth
//! for data the compiler never reasons about.

/// The generated table, laid out as documented in `scripts/generate-nid-names.py`.
static TABLE: &[u8] = include_bytes!("nid_names.bin");

const MAGIC: &[u8] = b"REMUNID1";
const HEADER_SIZE: usize = 16;
/// Eleven NID bytes, then the name's length, then a pad byte, then its offset into the blob.
const RECORD_SIZE: usize = 16;
const NID_LENGTH: usize = 11;

/// Returns the record count and the byte offset of the name blob, or `None` if the table is
/// missing, truncated, or does not describe itself consistently.
fn header() -> Option<(usize, usize)> {
    if !TABLE.starts_with(MAGIC) {
        return None;
    }
    let count = u32::from_le_bytes(TABLE.get(8..12)?.try_into().ok()?) as usize;
    let names_offset = u32::from_le_bytes(TABLE.get(12..16)?.try_into().ok()?) as usize;
    let index_end = HEADER_SIZE.checked_add(count.checked_mul(RECORD_SIZE)?)?;
    (index_end <= names_offset && names_offset <= TABLE.len()).then_some((count, names_offset))
}

/// Returns the NID a record is stored under.
fn record_nid(index: usize) -> Option<&'static str> {
    let start = HEADER_SIZE + index * RECORD_SIZE;
    std::str::from_utf8(TABLE.get(start..start + NID_LENGTH)?).ok()
}

/// Returns the name a record points at, bounds-checked against the blob.
fn record_name(index: usize, names_offset: usize) -> Option<&'static str> {
    let start = HEADER_SIZE + index * RECORD_SIZE;
    let length = *TABLE.get(start + NID_LENGTH)? as usize;
    let offset = u32::from_le_bytes(TABLE.get(start + 12..start + 16)?.try_into().ok()?) as usize;
    let begin = names_offset.checked_add(offset)?;
    std::str::from_utf8(TABLE.get(begin..begin.checked_add(length)?)?).ok()
}

/// Returns the symbol name recovered for `nid`, if the generator recovered one.
#[must_use]
pub fn generated_nid_name(nid: &str) -> Option<&'static str> {
    let (count, names_offset) = header()?;
    let (mut low, mut high) = (0, count);
    while low < high {
        let middle = low + (high - low) / 2;
        match record_nid(middle)?.cmp(nid) {
            std::cmp::Ordering::Less => low = middle + 1,
            std::cmp::Ordering::Greater => high = middle,
            std::cmp::Ordering::Equal => return record_name(middle, names_offset),
        }
    }
    None
}

/// Returns how many names the table carries, for coverage reporting.
#[cfg(test)]
#[must_use]
pub fn generated_nid_name_count() -> usize {
    header().map_or(0, |(count, _)| count)
}

#[cfg(test)]
mod tests {
    use super::{generated_nid_name, generated_nid_name_count, header, record_name, record_nid};
    use crate::sce_symbol_nid;

    #[test]
    fn the_table_parses_and_carries_names() {
        assert!(
            header().is_some(),
            "generated table is missing or malformed"
        );
        assert!(
            generated_nid_name_count() > 1000,
            "recovered only {} names",
            generated_nid_name_count()
        );
    }

    #[test]
    fn records_are_sorted_so_binary_search_reaches_every_one_of_them() {
        let (count, names_offset) = header().expect("table parses");
        let mut previous = "";
        for index in 0..count {
            let nid = record_nid(index).expect("record carries a NID");
            assert!(
                nid > previous,
                "table is unsorted at record {index}: {nid:?} follows {previous:?}"
            );
            previous = nid;
            let name = record_name(index, names_offset).expect("record carries a name");
            assert!(!name.is_empty(), "record {index} has an empty name");
            assert_eq!(generated_nid_name(nid), Some(name));
        }
    }

    /// The generator hashes names in Python; this checks it agrees with the runtime's own hashing,
    /// so a table built by a future edit to the script cannot quietly mislabel every import.
    #[test]
    fn every_recovered_name_hashes_back_to_the_nid_it_is_stored_under() {
        let (count, names_offset) = header().expect("table parses");
        for index in 0..count {
            let nid = record_nid(index).expect("record carries a NID");
            let name = record_name(index, names_offset).expect("record carries a name");
            assert_eq!(
                sce_symbol_nid(name),
                nid,
                "name {name:?} does not hash to {nid:?}"
            );
        }
    }

    /// Recovered names must never take precedence over, or disagree with, the hand-written table.
    #[test]
    fn the_generated_table_does_not_contradict_the_hand_maintained_one() {
        let (count, names_offset) = header().expect("table parses");
        for index in 0..count {
            let nid = record_nid(index).expect("record carries a NID");
            if let Some(known) = crate::known_nid_name(nid) {
                let generated = record_name(index, names_offset).expect("record carries a name");
                assert_eq!(
                    known, generated,
                    "generated name for {nid:?} disagrees with the hand-maintained table"
                );
            }
        }
    }

    #[test]
    fn a_nid_the_generator_never_saw_reports_no_name() {
        assert_eq!(generated_nid_name("AAAAAAAAAAA"), None);
    }

    #[test]
    fn a_lookup_that_is_not_a_nid_is_rejected_rather_than_matching_on_a_prefix() {
        assert_eq!(generated_nid_name(""), None);
        assert_eq!(generated_nid_name("AAAAAAAAAAAAAAAAAAAAAA"), None);
    }
}
