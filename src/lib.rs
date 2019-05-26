// Copyright (c) 2019 Brandon Thomas <bt@brand.io>, <echelon@gmail.com>
//
// #![deny(dead_code)]
// #![deny(missing_docs)]
// #![deny(unreachable_patterns)]
// #![deny(unused_extern_crates)]
// #![deny(unused_imports)]
// #![deny(unused_qualifications)]
//
//! Vocode.rs

#[cfg(test)] #[macro_use] extern crate expectest;

#[macro_use]
extern crate approx; // For the macro relative_eq!

extern crate nalgebra as na;
extern crate byteorder;

use std::io;
use std::fs::File;
use std::io::Cursor;

use byteorder::ReadBytesExt;
use byteorder::BigEndian;
use byteorder::LittleEndian;

use na::{U2, U3, Dynamic, MatrixArray, MatrixVec};
use na::Vector3;
use na::zero;
use na::VecStorage;
use na::Rotation3;
use na::Matrix;


/// Library version string
pub const VERSION_STRING : &'static str = "0.0.1";

pub fn test() {
  let axis  = Vector3::x_axis();
  let angle = 1.57;
  let b     = Rotation3::from_axis_angle(&axis, angle);

  relative_eq!(b.axis().unwrap(), axis);
  relative_eq!(b.angle(), angle);
}

pub fn read_mel_file() -> io::Result<()> {
  // https://en.cppreference.com/w/cpp/language/types
  // Dynamically sized and dynamically allocated matrix with
  // two rows and using 32-bit signed integers.
  type DMatrixf32 = Matrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic, Dynamic>>;

  let mut file = File::open("./matrices/mels/output.cpp-4.mel")?;

  //let mut reader = Cursor::new(file);

  let rows = file.read_i32::<LittleEndian>()?;
  let cols = file.read_i32::<LittleEndian>()?;

  let mut mat : DMatrixf32 = Matrix::<f32, Dynamic, Dynamic, _>::zeros(rows as usize, cols as usize);
  println!("Rows: {}, Cols: {}", rows, cols);

  for (i, mut cell) in mat.iter_mut().enumerate() {
    let data = file.read_f32::<LittleEndian>()?;
    *cell = data;
  }

  for (i, mut cell) in mat.iter().enumerate() {
    println!("i: {}, data: {}", i, cell);
  }


  Ok(())
}


#[cfg(test)]
mod tests {
  use super::*;
  use expectest::prelude::*;

  #[test]
  fn test() {
    expect!("".chars()).to(be_empty());
  }

  #[test]
  fn test_2() {
    expect!("".chars()).to(be_empty());
  }

  #[test]
  fn test_3() {
    expect!(read_mel_file()).to(be_ok());
  }
}