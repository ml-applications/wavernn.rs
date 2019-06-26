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

#[macro_use]
extern crate approx;
extern crate byteorder;
#[cfg(test)] #[macro_use] extern crate expectest;
extern crate itertools;
extern crate nalgebra as na_do_not_use;
#[macro_use(array)]
extern crate ndarray;

// For the macro relative_eq!

use std::fs::File;
use std::io;
use std::io::{Cursor, Read};
use std::io::Error as IoError;

use byteorder::BigEndian;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use itertools::enumerate;
use ndarray::Array;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::ArrayBase;
use ndarray::ArrayD;
use ndarray::Dim;
use ndarray::Ix0;
use ndarray::Ix1;
use ndarray::Ix2;
use ndarray::IxDyn;

use layers::Conv1dLayer;
use layers::Conv2dLayer;
use parser::read_model_file;

//use ::LayerType::Conv1d;

mod layers;
mod parser;

/// Library version string
pub const VERSION_STRING : &'static str = "0.0.1";

pub fn read_mel_file(filename: &str) -> io::Result<()> {
  use na_do_not_use::{U2, U3, Dynamic, MatrixArray, MatrixVec};
  use na_do_not_use::Vector3;
  use na_do_not_use::zero;
  use na_do_not_use::VecStorage;
  use na_do_not_use::Rotation3;
  use na_do_not_use::Matrix;

  // https://en.cppreference.com/w/cpp/language/types
  // Dynamically sized and dynamically allocated matrix with
  // two rows and using 32-bit signed integers.
  type DMatrixf32 = Matrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic, Dynamic>>;

  let mut file = File::open(filename)?;

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
  use expectest::prelude::*;

  use super::*;

  #[test]
  fn test_read_mel_file() {
    let filename = "./matrices/mels/output.cpp-4.mel";
    expect!(read_mel_file(filename)).to(be_ok());
  }

  #[test]
  fn test_read_model() {
    let filename = "./matrices/model/model.bin";
    expect!(read_model_file(filename)).to(be_ok());
  }
}