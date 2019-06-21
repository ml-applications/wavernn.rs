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
#[macro_use(array)]
extern crate ndarray;
extern crate byteorder;
extern crate itertools;

use std::io;
use std::fs::File;
use std::io::{Cursor, Read};
use std::io::Error as IoError;

use byteorder::ReadBytesExt;
use byteorder::BigEndian;
use byteorder::LittleEndian;
use itertools::enumerate;

use na::{U2, U3, Dynamic, MatrixArray, MatrixVec};
use na::Vector3;
use na::zero;
use na::VecStorage;
use na::Rotation3;
use na::Matrix;
use ndarray::{ArrayBase, Array, Dim, Ix2, Ix1, Ix0, Array2, Array1};
//use ::LayerType::Conv1d;

mod layers;

use layers::Conv1dLayer;
use layers::Conv2dLayer;
use layers::read_model_file;

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

  // for (i, mut cell) in mat.iter().enumerate() {
  //   println!("i: {}, data: {}", i, cell);
  // }

  Ok(())
}
/*
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int useBias;
        int inChannels;
        int outChannels;
        int kernelSize;
    };

    std::vector<Matrixf> weight;
    Vectorf bias;

    bool hasBias;
    int inChannels;
    int outChannels;
    int nKernel;
*/

/*impl Conv1dLayer {
  fn parse(file: &mut File) -> io::Result<Conv1dLayer> {
    let el_size = file.read_i32::<LittleEndian>()?;
    let use_bias = file.read_i32::<LittleEndian>()?;
    let in_channels = file.read_i32::<LittleEndian>()?;
    let out_channels = file.read_i32::<LittleEndian>()?;
    let kernel_size = file.read_i32::<LittleEndian>()?;

    if el_size != 2 && el_size != 4 {
      return Err(IoError::from_raw_os_error(0)); // TODO: Actual error
    }

    let use_bias = match use_bias {
      0 => false,
      _ => true,
    };

    print!("out_channels: {}", out_channels);

    //let weights : Vec<NDA> = Vec::with_capacity(out_channels as usize);

    if kernel_size == 1 {
      // If kernel is 1x then convolution is just matrix multiplication.
      // Load weight into the first element and handle separately.

    } else {
      println!("test");
      // std::vector<Matrixf> weight;
      let mut weights : Vec<Array2<f32>> = Vec::with_capacity(out_channels as usize);

      for i in 0 .. out_channels as usize {
        let mut weight = Array2::<f32>::zeros((in_channels as usize, out_channels as usize));

        for (j, element) in enumerate(&mut weight) {
          println!("i: {}", j);
          // TODO: THis is what is rbeaking. I'm reading too much or too little or something.
          *element = file.read_f32::<LittleEndian>().expect("This should work");
        }


        weights.push(weight);
      }

      fn f(array: &Array2<f32>) {
        println!("{:?}", array);
      }

      for w in weights {
        f(&w);
      }
    }

    if use_bias {

    }

    Err(IoError::from_raw_os_error(0)) // TODO: Actual error

    /*Ok(Conv1dLayer {
      el_size,
      use_bias,
      in_channels,
      out_channels,
      kernel_size,
    })*/
  }
}*/

#[cfg(test)]
mod tests {
  use super::*;
  use expectest::prelude::*;

  #[test]
  fn test_1() {
    expect!(read_mel_file()).to(be_ok());
  }

  #[test]
  fn test_2() {
    let filename = "./matrices/model/model.bin";
    expect!(read_model_file(filename)).to(be_ok());
  }
}