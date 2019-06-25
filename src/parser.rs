use byteorder::BigEndian;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use itertools::enumerate;
use na::Matrix;
use na::Rotation3;
use na::VecStorage;
use na::Vector3;
use na::zero;
use na::{U2, U3, Dynamic, MatrixArray, MatrixVec};
use ndarray::{ArrayBase, Array, Dim, Ix2, Ix1, Ix0, Array2, Array1};
use std::fs::File;
use std::io::Error as IoError;
use std::io::{Cursor, Read};
use std::io;

use layers::*;

/**
 * Read an ML model file.
 * See net_impl.cpp : Model::loadNext(FILE *fd)
 */
pub fn read_model_file(filename: &str) -> io::Result<()> {
  type DMatrixf32 = Matrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic, Dynamic>>;

  let mut file = File::open(filename)?;

  let header = ModelHeader::parse(&mut file)?;

  println!("model_header: {:?}", header);

  let resnet = Resnet::parse(&mut file)?;

  println!("resnet : {:?}", header);

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

pub trait ParseStruct<T> {
  fn parse(file: &mut File) -> io::Result<T>;
}

impl ParseStruct<ModelHeader> for ModelHeader {
  fn parse(file: &mut File) -> io::Result<ModelHeader> {
    Ok( ModelHeader {
      num_res_blocks: file.read_i32::<LittleEndian>()?,
      num_upsample: file.read_i32::<LittleEndian>()?,
      total_scale: file.read_i32::<LittleEndian>()?,
      n_pad: file.read_i32::<LittleEndian>()?,
    })
  }
}

impl ParseStruct<Resnet> for Resnet {
  fn parse(file: &mut File) -> io::Result<Resnet> {
    Ok( Resnet {
      conv_in: TorchLayer::parse(file)?,
      batch_norm: TorchLayer::parse(file)?,
      resblock: ResBlock::parse(file)?,
      conv_out: TorchLayer::parse(file)?,
      stretch2d: TorchLayer::parse(file)?,
    })
  }
}

impl ParseStruct<TorchLayer> for TorchLayer {
  fn parse(file: &mut File) -> io::Result<TorchLayer> {
    let header = TorchLayerHeader::parse(file)?;
    unimplemented!()
  }
}

impl ParseStruct<TorchLayerHeader> for TorchLayerHeader {
  fn parse(file: &mut File) -> io::Result<TorchLayerHeader> {
    let layer_type = file.read_i32::<LittleEndian>()?;

    let layer_type = match layer_type {
      1 => LayerType::Conv1d,
      2 => LayerType::Conv2d,
      3 => LayerType::BatchNorm1d,
      4 => LayerType::Linear,
      5 => LayerType::GRU,
      6 => LayerType::Stretch2d,
      _ => return Err(IoError::from_raw_os_error(0)), // TODO: Actual error
    };

    Ok (TorchLayerHeader {
      layer_type,
      name: read_name(file)?,
    })
  }
}


impl ParseStruct<ResBlock> for ResBlock {
  fn parse(file: &mut File) -> io::Result<ResBlock> {
    unimplemented!()
  }
}

impl ParseStruct<Conv1dLayer> for Conv1dLayer {
  fn parse(file: &mut File) -> io::Result<Conv1dLayer> {
    let el_size = file.read_i32::<LittleEndian>()?;
    let use_bias = file.read_i32::<LittleEndian>()?;
    let in_channels = file.read_i32::<LittleEndian>()?;
    let out_channels = file.read_i32::<LittleEndian>()?;
    let kernel_size = file.read_i32::<LittleEndian>()?;

    if el_size != 2 && el_size != 4 {
      return Err(IoError::from_raw_os_error(0)); // TODO: Actual error
    }

    let has_bias = match use_bias {
      0 => false,
      _ => true,
    };

    if kernel_size == 1 {
      // TODO
    } else {
      // TODO
      let mut weights : Vec<Array2<f32>> = Vec::with_capacity(out_channels as usize);

      for i in 0 .. out_channels as usize {
        let mut weight = Array2::<f32>::zeros((in_channels as usize, out_channels as usize));

        for (j, element) in enumerate(&mut weight) {
          //println!("i: {}", j);
          // TODO: THis is what is breaking. I'm reading too much or too little or something.
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

    let mut bias= Vec::new();

    if has_bias {
      bias.reserve(kernel_size as usize);
      read_into_matrix(file, &mut bias);
    }

    Ok(Self {
      weight: Vec::new(), // TODO
      bias,
      has_bias,
      in_channels,
      out_channels,
      n_kernel: kernel_size,
    })
  }
}

fn read_into_matrix(file: &mut File, mat: &mut Vec<f32>) -> io::Result<()> {
  for (j, element) in enumerate(mat) {
    *element = file.read_f32::<LittleEndian>().expect("This should work");
  }
  Ok(())
}

fn read_name(file: &mut File) -> io::Result<String> {
  let mut buffer = [0; 64];
  //let mut buffer = String::with_capacity(64usize);
  file.read_exact(&mut buffer);

  let name = String::from_utf8_lossy(&buffer);

  // TODO: Remove trailing null bytes.

  Ok(name.into())
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
