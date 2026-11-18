//! Units of NGFF axes (UDUNITS-2 names), with ozp's table (`spatial.unit_factor`).

/// The size of a unit in its base unit (metre or second).
const UNIT_SIZE: [(&str, f64, &str); 17] = [
    ("angstrom", 1e-10, "m"),
    ("picometer", 1e-12, "m"),
    ("nanometer", 1e-9, "m"),
    ("micrometer", 1e-6, "m"),
    ("millimeter", 1e-3, "m"),
    ("centimeter", 1e-2, "m"),
    ("decimeter", 1e-1, "m"),
    ("meter", 1.0, "m"),
    ("kilometer", 1e3, "m"),
    ("picosecond", 1e-12, "s"),
    ("nanosecond", 1e-9, "s"),
    ("microsecond", 1e-6, "s"),
    ("millisecond", 1e-3, "s"),
    ("second", 1.0, "s"),
    ("minute", 60.0, "s"),
    ("hour", 3600.0, "s"),
    ("day", 86400.0, "s"),
];

fn size(unit: &str) -> Option<(f64, &'static str)> {
    UNIT_SIZE
        .iter()
        .find(|(n, _, _)| *n == unit)
        .map(|(_, s, b)| (*s, *b))
}

/// How many `into` one `unit` is (micrometer into nanometer: 1000); 1 for the same unit
/// or two missing ones; None when they cannot be converted (an unknown unit, a length
/// and a time, one missing).
pub fn unit_factor(unit: Option<&str>, into: Option<&str>) -> Option<f64> {
    if unit == into {
        return Some(1.0);
    }
    let (a, base_a) = size(unit?)?;
    let (b, base_b) = size(into?)?;
    if base_a != base_b {
        return None;
    }
    Some(a / b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn factors_as_ozp() {
        // computed as ozp does (1e-6 / 1e-9 = 999.9999999999999 in both)
        assert_eq!(
            unit_factor(Some("micrometer"), Some("nanometer")),
            Some(1e-6 / 1e-9)
        );
        assert_eq!(unit_factor(None, None), Some(1.0));
        assert_eq!(unit_factor(Some("weird"), Some("weird")), Some(1.0));
        assert_eq!(unit_factor(Some("micrometer"), Some("second")), None);
        assert_eq!(unit_factor(Some("micrometer"), None), None);
        assert_eq!(unit_factor(Some("weird"), Some("meter")), None);
        assert_eq!(unit_factor(Some("minute"), Some("second")), Some(60.0));
    }
}
