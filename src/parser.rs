use std::fs::File;
use std::io;
use std::io::Cursor;
use std::io::Error as IoError;
use std::io::Read;

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

use layers::*;

/**
 * Read an ML model file.
 * See net_impl.cpp : Model::loadNext(FILE *fd)
 */
pub fn read_model_file(filename: &str) -> io::Result<()> {
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
    println!("TorchLayerHeader: {:?}", header);

    let result = match header.layer_type {
      LayerType::Conv1d => TorchLayer::Conv1dLayer(Conv1dLayer::parse(file)?),
      LayerType::Conv2d => TorchLayer::Conv2dLayer(Conv2dLayer::parse(file)?),
      LayerType::BatchNorm1d => TorchLayer::BatchNorm1dLayer(BatchNorm1dLayer::parse(file)?),
      LayerType::Linear => TorchLayer::LinearLayer(LinearLayer::parse(file)?),
      LayerType::GRU => TorchLayer::GruLayer(GruLayer::parse(file)?),
      LayerType::Stretch2d => TorchLayer::Stretch2dLayer(Stretch2dLayer::parse(file)?),
    };

    Ok(result)
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
    println!("ResBlock.parse()");
    unimplemented!()
  }
}

impl ParseStruct<Conv1dLayer> for Conv1dLayer {
  fn parse(file: &mut File) -> io::Result<Conv1dLayer> {
    println!("Conv1dLayer.parse()");

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

    let mut weight = Vec::new();

    if kernel_size == 1 {
      // If kernel is 1x then convolution is just matrix multiplication.
      // Load weight into the first element and handle separately.
      let shape = [in_channels as usize, out_channels as usize];
      let matrix = ArrayD::<f32>::zeros(IxDyn(&shape));
      weight.push(matrix);

      // TODO
    } else {
      // TODO
      /*let mut weights : Vec<Array2<f32>> = Vec::with_capacity(out_channels as usize);

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
      }*/
    }

    let mut bias= Vec::new();

    if has_bias {
      bias.reserve(kernel_size as usize);
      read_into_matrix(file, &mut bias);
    }

    Ok(Conv1dLayer {
      weight,
      bias,
      has_bias,
      in_channels,
      out_channels,
      n_kernel: kernel_size,
    })
  }
}

impl ParseStruct<Conv2dLayer> for Conv2dLayer {
  fn parse(file: &mut File) -> io::Result<Conv2dLayer> {
    println!("Conv2dLayer.parse()");

    let el_size = file.read_i32::<LittleEndian>()?;
    let n_kernel = file.read_i32::<LittleEndian>()?;

    if el_size != 2 && el_size != 4 {
      return Err(IoError::from_raw_os_error(0)); // TODO: Actual error
    }

    let mut weight= Vec::with_capacity(n_kernel as usize);

    read_into_matrix(file, &mut weight);

    Ok(Conv2dLayer {
      weight,
      n_kernel,
    })
  }
}

impl ParseStruct<BatchNorm1dLayer> for BatchNorm1dLayer {
  fn parse(file: &mut File) -> io::Result<BatchNorm1dLayer> {
    let el_size = file.read_i32::<LittleEndian>()?;
    let in_channels= file.read_i32::<LittleEndian>()?;
    let eps = file.read_f32::<LittleEndian>()?;

    if el_size != 2 && el_size != 4 {
      return Err(IoError::from_raw_os_error(0)); // TODO: Actual error
    }

    let mut weight= Vec::with_capacity(in_channels as usize);
    let mut bias= Vec::with_capacity(in_channels as usize);
    let mut running_mean= Vec::with_capacity(in_channels as usize);
    let mut running_var= Vec::with_capacity(in_channels as usize);

    read_into_matrix(file, &mut weight);
    read_into_matrix(file, &mut bias);
    read_into_matrix(file, &mut running_mean);
    read_into_matrix(file, &mut running_var);

    Ok(BatchNorm1dLayer {
      weight,
      bias,
      running_mean,
      running_var,
      eps,
      n_channels: in_channels,
    })
  }
}

impl ParseStruct<LinearLayer> for LinearLayer {
  fn parse(file: &mut File) -> io::Result<LinearLayer> {
    // Read header
    let el_size = file.read_i32::<LittleEndian>()?;
    let n_rows = file.read_i32::<LittleEndian>()?;
    let n_cols = file.read_i32::<LittleEndian>()?;

    if el_size != 2 && el_size != 4 {
      return Err(IoError::from_raw_os_error(0)); // TODO: Actual error
    }

    // TODO: CompMatrix mat;

    let mut bias = Vec::with_capacity(n_rows as usize);

    read_into_matrix(file, &mut bias);

    Ok(Self {
      bias,
      n_rows,
      n_cols,
    })
  }
}

impl ParseStruct<GruLayer> for GruLayer {
  fn parse(file: &mut File) -> io::Result<GruLayer> {
    // Read header
    let el_size = file.read_i32::<LittleEndian>()?;
    let n_hidden = file.read_i32::<LittleEndian>()?;
    let n_input = file.read_i32::<LittleEndian>()?;

    if el_size != 2 && el_size != 4 {
      return Err(IoError::from_raw_os_error(0)); // TODO: Actual error
    }

    let n_rows = n_hidden;
    let n_cols = n_input;

    // TODO: CompMatrix W_ir,W_iz,W_in;
    // TODO: CompMatrix W_hr,W_hz,W_hn;

    let mut b_ir = Vec::with_capacity(n_hidden as usize);
    let mut b_iz = Vec::with_capacity(n_hidden as usize);
    let mut b_in = Vec::with_capacity(n_hidden as usize);

    read_into_matrix(file, &mut b_ir);
    read_into_matrix(file, &mut b_iz);
    read_into_matrix(file, &mut b_in);

    let mut b_hr = Vec::with_capacity(n_hidden as usize);
    let mut b_hz = Vec::with_capacity(n_hidden as usize);
    let mut b_hn = Vec::with_capacity(n_hidden as usize);

    read_into_matrix(file, &mut b_hr);
    read_into_matrix(file, &mut b_hz);
    read_into_matrix(file, &mut b_hn);

    Ok(GruLayer {
      b_ir,
      b_iz,
      b_in,
      b_hr,
      b_hz,
      b_hn,
      n_rows,
      n_cols,
    })
  }
}

impl ParseStruct<Stretch2dLayer> for Stretch2dLayer {
  fn parse(file: &mut File) -> io::Result<Stretch2dLayer> {
    Ok(Stretch2dLayer {
      x_scale: file.read_i32::<LittleEndian>()?,
      y_scale: file.read_i32::<LittleEndian>()?,
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

