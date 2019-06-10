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
use ::LayerType::Conv1d;

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
class Model{
  struct  Header{
    int num_res_blocks;
    int num_upsample;
    int total_scale;
    int nPad;
  };
  Header header;

  UpsampleNetwork upsample;
  Resnet resnet;
  TorchLayer I;
  TorchLayer rnn1;
  TorchLayer fc1;
  TorchLayer fc2;
*/
pub fn read_model_file() -> io::Result<()> {
  type DMatrixf32 = Matrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic, Dynamic>>;

  /*
  From C++,
  Header.num_res_blocks ...3
  Header.num_upsample...3
  Header.total_scale ...200
  Header.npad...2
  */
  let mut file = File::open("./matrices/model/model.bin")?;

  //let mut reader = Cursor::new(file);

  let num_res_blocks = file.read_i32::<LittleEndian>()?;
  let num_upsample = file.read_i32::<LittleEndian>()?;
  let total_scale = file.read_i32::<LittleEndian>()?;
  let n_pad = file.read_i32::<LittleEndian>()?;

  println!("num_res_blocks: {}", num_res_blocks);
  println!("num_upsample: {}", num_upsample);
  println!("total_scale: {}", total_scale);
  println!("n_pad: {}", n_pad);

  /*
  class Resnet {
    TorchLayer conv_in;
    TorchLayer batch_norm;
    ResBlock resblock;
    TorchLayer conv_out;
    TorchLayer stretch2d;  //moved stretch2d layer into resnet from upsample as in python code
  }
  */

  //TorchLayer::Header header;
  //fread(&header, sizeof(TorchLayer::Header), 1, fd);
  /*

    struct  Header{
        //int size; //size of data blob, not including this header
        enum class LayerType : int { Conv1d=1, Conv2d=2, BatchNorm1d=3, Linear=4, GRU=5, Stretch2d=6 } layerType;
        char name[64]; //layer name for debugging
    };*/

  let layer_type = file.read_i32::<LittleEndian>()?;

  let layer_type = parse_layer_type(layer_type).unwrap();

  println!("layer_type: {:?}", layer_type);

  /*let mut buffer = [0; 64];
  file.read_exact(&mut buffer);
  let name = String::from_utf8_lossy(&buffer);*/

  let name = read_name(&mut file).unwrap();

  println!("name: {}", name);

  match (layer_type) {
    LayerType::Conv1d => {
      let layer = Conv1dLayer::parse(&mut file)?;
      println!("Layer: {:?}", layer);
    },
    LayerType::Conv2d => {},
    LayerType::BatchNorm1d => {},
    LayerType::Linear => {},
    LayerType::GRU => {},
    LayerType::Stretch2d => {},
  }

  Ok(())
}

#[derive(Clone,Copy,Debug)]
enum LayerType {
  Conv1d,
  Conv2d,
  BatchNorm1d,
  Linear,
  GRU,
  Stretch2d,
}

// TODO: Use From<> trait.
fn parse_layer_type(layer_type: i32) -> Option<LayerType> {
  match layer_type {
    1 => Some(LayerType::Conv1d),
    2 => Some(LayerType::Conv2d),
    3 => Some(LayerType::BatchNorm1d),
    4 => Some(LayerType::Linear),
    5 => Some(LayerType::GRU),
    6 => Some(LayerType::Stretch2d),
    _ => None,
  }
}

fn read_name(file: &mut File) -> Option<String> {
  let mut buffer = [0; 64];
  //let mut buffer = String::with_capacity(64usize);
  file.read_exact(&mut buffer);

  let name = String::from_utf8_lossy(&buffer);

  // TODO: Remove trailing null bytes.

  Some(name.into())
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

#[derive(Debug)]
struct Conv1dLayer {
  el_size: i32,
  use_bias: bool,
  in_channels: i32,
  out_channels: i32,
  kernel_size: i32,
}

impl Conv1dLayer {
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
}

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
    expect!(read_model_file()).to(be_ok());
  }
}