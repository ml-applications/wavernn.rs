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
pub fn read_model_file(filename: &str) -> io::Result<Model> {
  let mut file = File::open(filename)?;

  let header = ModelHeader::parse(&mut file)?;

  println!("model_header: {:?}", header);

  let resnet = Resnet::parse(&mut file)?;

  println!("resnet : {:?}", header);

  let upsample = UpsampleNetwork::parse(&mut file)?;

  println!("upsample : {:?}", upsample);

  let i = TorchLayer::parse(&mut file)?;
  let rnn1 = TorchLayer::parse(&mut file)?;
  let fc1= TorchLayer::parse(&mut file)?;
  let fc2= TorchLayer::parse(&mut file)?;

  Ok(Model {
    header,
    upsample,
    resnet,
    i,
    rnn1,
    fc1,
    fc2,
  })
}

pub trait ParseStruct<T> {
  fn parse(file: &mut File) -> io::Result<T>;
}

impl ParseStruct<ModelHeader> for ModelHeader {
  fn parse(file: &mut File) -> io::Result<ModelHeader> {
    println!("ModelHeader.parse()");
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
    println!("Resnet.parse()");
    Ok( Resnet {
      conv_in: TorchLayer::parse(file)?,
      batch_norm: TorchLayer::parse(file)?,
      resblock: ResBlock::parse(file)?,
      conv_out: TorchLayer::parse(file)?,
      stretch2d: TorchLayer::parse(file)?,
    })
  }
}

impl CompMatrix {
  fn parse_matrix(file: &mut File, el_size: i32, n_rows: i32, n_cols: i32) -> io::Result<CompMatrix> {
    panic!("TODO: THIS IS WHERE I LEFT OFF. COMPMATRIX");
    println!("CompMatrix.parse_matrix()");
    Ok( CompMatrix {
      weight: Vec::new(), // TODO
      row_idx: Vec::new(), // TODO
      col_idx: Vec::new(), // TODO
      n_groups: 1234, // TODO
      n_rows,
      n_cols,
    })
  }
}

impl ParseStruct<UpsampleNetwork> for UpsampleNetwork {
  fn parse(file: &mut File) -> io::Result<UpsampleNetwork> {
    println!("UpsampleNetwork.parse()");

    // const int UPSAMPLE_LAYERS = 3;
    let mut up_layers = Vec::with_capacity(3 * 2);

    for _i in 0 .. 6 {
      let layer = TorchLayer::parse(file)?;
      up_layers.push(layer);
    }

    Ok(UpsampleNetwork {
      up_layers,
    })
  }
}


impl ParseStruct<TorchLayer> for TorchLayer {
  fn parse(file: &mut File) -> io::Result<TorchLayer> {
    println!("TorchLayer.parse()");
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
    println!("TorchLayerHeader.parse()");
    let layer_type = file.read_i32::<LittleEndian>()?;

    println!(" -> layer_type = {}", layer_type);

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
    // const int RES_BLOCKS = 3;
    let mut resblock = Vec::with_capacity(3 * 4);

    for i in 0 .. 12 {
      println!(" - ResBlock.parse #{}", i);
      let layer = TorchLayer::parse(file)?;
      resblock.push(layer);
    }

    Ok(ResBlock {
      resblock,
    })
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

    println!("-> el_size {}", el_size);
    println!("-> use_bias {}", use_bias);
    println!("-> in_channels {}", in_channels);
    println!("-> out_channels {}", out_channels);
    println!("-> kernel_size {}", kernel_size);

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
      let mut matrix = ArrayD::<f32>::zeros(IxDyn(&shape));

      for _i in 0 .. in_channels {
        for _j in 0 .. out_channels {
          let _element = file.read_f32::<LittleEndian>()?;
          // TODO: SET
        }
      }
      weight.push(matrix);
    } else {
      let shape = [in_channels as usize, kernel_size as usize];

      for i in 0 .. out_channels as usize {
        let matrix = ArrayD::<f32>::zeros(IxDyn(&shape));
        for _j in 0 .. in_channels {
          for _k in 0 .. kernel_size {
            let _element = file.read_f32::<LittleEndian>()?;
            // TODO: SET
          }
        }
        weight.push(matrix);
      }
    }

    println!("-> has bias? {}", has_bias);

    let bias = if has_bias {
      read_vec_f32(file, out_channels as usize)?
    } else {
      Vec::new()
    };

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

    let weight = read_vec_f32(file, n_kernel as usize)?;

    Ok(Conv2dLayer {
      weight,
      n_kernel,
    })
  }
}

impl ParseStruct<BatchNorm1dLayer> for BatchNorm1dLayer {
  fn parse(file: &mut File) -> io::Result<BatchNorm1dLayer> {
    println!("BatchNorm1dLayer.parse()");

    let el_size = file.read_i32::<LittleEndian>()?;
    let in_channels= file.read_i32::<LittleEndian>()?;
    let eps = file.read_f32::<LittleEndian>()?;

    println!("> el_size = {}", el_size);
    println!("> in_channels = {}", in_channels);
    println!("> eps = {}", eps);

    if el_size != 2 && el_size != 4 {
      return Err(IoError::from_raw_os_error(0)); // TODO: Actual error
    }

    let channels = in_channels as usize;
    let weight = read_vec_f32(file, channels)?;
    let bias = read_vec_f32(file, channels)?;
    let running_mean = read_vec_f32(file, channels)?;
    let running_var = read_vec_f32(file, channels)?;

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
    println!("LinearLayer.parse()");
    // Read header
    let el_size = file.read_i32::<LittleEndian>()?;
    let n_rows = file.read_i32::<LittleEndian>()?;
    let n_cols = file.read_i32::<LittleEndian>()?;

    if el_size != 2 && el_size != 4 {
      return Err(IoError::from_raw_os_error(0)); // TODO: Actual error
    }

    // read compressed array
    let mat = CompMatrix::parse_matrix(file, el_size, n_cols, n_cols)?;

    let bias = read_vec_f32(file, n_rows as usize)?;

    Ok(Self {
      mat,
      bias,
      n_rows,
      n_cols,
    })
  }
}

impl ParseStruct<GruLayer> for GruLayer {
  fn parse(file: &mut File) -> io::Result<GruLayer> {
    println!("GruLayer.parse()");
    // Read header
    let el_size = file.read_i32::<LittleEndian>()?;
    let n_hidden = file.read_i32::<LittleEndian>()?;
    let n_input = file.read_i32::<LittleEndian>()?;

    if el_size != 2 && el_size != 4 {
      return Err(IoError::from_raw_os_error(0)); // TODO: Actual error
    }

    let n_rows = n_hidden;
    let n_cols = n_input;

    let w_ir = CompMatrix::parse_matrix(file, el_size, n_hidden, n_input)?;
    let w_iz = CompMatrix::parse_matrix(file, el_size, n_hidden, n_input)?;
    let w_in = CompMatrix::parse_matrix(file, el_size, n_hidden, n_input)?;

    let w_hr = CompMatrix::parse_matrix(file, el_size, n_hidden, n_hidden)?;
    let w_hz = CompMatrix::parse_matrix(file, el_size, n_hidden, n_hidden)?;
    let w_hn = CompMatrix::parse_matrix(file, el_size, n_hidden, n_hidden)?;

    let hidden = n_hidden as usize;

    let mut b_ir = read_vec_f32(file, hidden)?;
    let mut b_iz = read_vec_f32(file, hidden)?;
    let mut b_in = read_vec_f32(file, hidden)?;

    let mut b_hr = read_vec_f32(file, hidden)?;
    let mut b_hz = read_vec_f32(file, hidden)?;
    let mut b_hn = read_vec_f32(file, hidden)?;

    Ok(GruLayer {
      w_ir,
      w_iz,
      w_in,
      w_hr,
      w_hz,
      w_hn,
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
    println!("Stretch2dLayer.parse()");
    Ok(Stretch2dLayer {
      x_scale: file.read_i32::<LittleEndian>()?,
      y_scale: file.read_i32::<LittleEndian>()?,
    })
  }
}

fn read_vec_f32(file: &mut File, size: usize) -> io::Result<Vec<f32>> {
  let mut vec = Vec::with_capacity(size);
  for _i in 0 .. size {
    vec.push(file.read_f32::<LittleEndian>()?);
  }
  Ok(vec)
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

