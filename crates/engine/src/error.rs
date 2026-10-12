use std::fmt;

/// Everything that can go wrong loading weights or a tokenizer: the message says what and where.
#[derive(Debug, Clone, PartialEq)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;

/// `bail!("bad {thing}")` returns early with an `Error`.
macro_rules! bail {
    ($($arg:tt)*) => { return Err($crate::error::Error(format!($($arg)*))) };
}
pub(crate) use bail;
