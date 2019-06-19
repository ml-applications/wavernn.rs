use byteorder::BigEndian;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use std::fs::File;
use itertools::enumerate;
use std::io::Error as IoError;
use std::io;

pub struct TorchLayer_BaseLayer {
  /* class TorchLayer : public BaseLayer {
    struct  Header{
        //int size; //size of data blob, not including this header
        enum class LayerType : int { Conv1d=1, Conv2d=2, BatchNorm1d=3, Linear=4, GRU=5, Stretch2d=6 } layerType;
        char name[64]; //layer name for debugging
    };

    BaseLayer* impl;
  */
}

#[derive(Debug)]
pub struct Conv1dLayer {
  /* class Conv1dLayer : public TorchLayer{
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
  el_size: i32,
  use_bias: bool,
  in_channels: i32,
  out_channels: i32,
  kernel_size: i32,
}

#[derive(Debug)]
pub struct Conv2dLayer {
  /* class Conv2dLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nKernel;  //kernel size. special case of conv2d used in WaveRNN
    };
    Vectorf weight;
    int nKernel;
  */
  weight: Vec<f32>,
  n_kernel: i32,
}

#[derive(Debug)]
pub struct BatchNorm1dLayer {
  /* class BatchNorm1dLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int inChannels;
        float eps;
    };
    Vectorf weight;
    Vectorf bias;
    Vectorf running_mean;
    Vectorf running_var;
    float eps;
    int nChannels;
  */
}


#[derive(Debug)]
pub struct LinearLayer {
  /* class LinearLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nRows;
        int nCols;
    };
    CompMatrix mat;
    Vectorf bias;
    int nRows;
    int nCols;
  */
}

#[derive(Debug)]
pub struct GruLayer {
  /* class GRULayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nHidden;
        int nInput;
    };
    CompMatrix W_ir,W_iz,W_in;
    CompMatrix W_hr,W_hz,W_hn;
    Vectorf b_ir,b_iz,b_in;
    Vectorf b_hr,b_hz,b_hn;
    int nRows;
    int nCols;
  */

  // TODO: CompMatrix W_ir,W_iz,W_in;
  // TODO: CompMatrix W_hr,W_hz,W_hn;

  b_ir: Vec<f32>,
  b_iz: Vec<f32>,
  b_in: Vec<f32>,

  b_hr: Vec<f32>,
  b_hz: Vec<f32>,
  b_hn: Vec<f32>,

  n_rows: i32,
  n_cols: i32,
}

impl GruLayer {

  fn parse(file: &mut File) -> io::Result<Self> {
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

    Ok(Self {
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

#[derive(Debug)]
pub struct Stretch2dLayer {
  /* class Stretch2dLayer : public TorchLayer{
    struct  Header{
        int x_scale;
        int y_scale;
    };
    int x_scale;
    int y_scale;
  */

  x_scale: i32,
  y_scale: i32,
}

impl Stretch2dLayer {

  fn parse(file: &mut File) -> io::Result<Self> {
    Ok(Self {
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