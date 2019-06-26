use byteorder::BigEndian;
use byteorder::LittleEndian;
use byteorder::ReadBytesExt;
use itertools::enumerate;
use ndarray::{ArrayBase, Array, Dim, Ix2, Ix1, Ix0, Array2, Array1};
use std::fs::File;
use std::io::Error as IoError;
use std::io::{Cursor, Read};
use std::io;

use parser::ParseStruct;

/* class TorchLayer : public BaseLayer {
  struct  Header{
      //int size; //size of data blob, not including this header
      enum class LayerType : int { Conv1d=1, Conv2d=2, BatchNorm1d=3, Linear=4, GRU=5, Stretch2d=6 } layerType;
      char name[64]; //layer name for debugging
  };

  BaseLayer* impl;
*/

//pub type DMatrixf32 = Matrix<f32, Dynamic, Dynamic, VecStorage<f32, Dynamic, Dynamic>>;

#[derive(Clone,Debug)]
pub enum TorchLayer {
  Conv1dLayer(Conv1dLayer),
  Conv2dLayer(Conv2dLayer),
  BatchNorm1dLayer(BatchNorm1dLayer),
  LinearLayer(LinearLayer),
  GruLayer(GruLayer),
  Stretch2dLayer(Stretch2dLayer),
}

#[derive(Clone,Debug)]
pub struct TorchLayerHeader {
  pub layer_type: LayerType,
  pub name: String,
}

// net_impl.h
#[derive(Clone,Debug)]
pub struct ModelHeader {
  pub num_res_blocks: i32,
  pub num_upsample: i32,
  pub total_scale: i32,
  pub n_pad: i32,
}

#[derive(Clone,Debug)]
pub struct UpsampleNetwork {
  pub up_layers: Vec<TorchLayer>,
}

#[derive(Clone,Debug)]
pub struct ResBlock {
  pub resblock: Vec<TorchLayer>,
}

#[derive(Clone,Debug)]
pub struct Resnet {
  pub conv_in: TorchLayer,
  pub batch_norm: TorchLayer,
  pub resblock: ResBlock,
  pub conv_out: TorchLayer,
  pub stretch2d: TorchLayer,
}

#[derive(Clone,Debug)]
pub struct Model {
  /*class Model{
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
  pub header: ModelHeader,
  pub upsample: UpsampleNetwork,
  pub resnet: Resnet,
  pub i: TorchLayer,
  pub rnn1: TorchLayer,
  pub fc1: TorchLayer,
  pub fc2: TorchLayer,
}

// wavernn.h
#[derive(Clone,Debug)]
pub struct CompMatrix {
  pub weight: Vec<f32>, // TODO: Is this right? Jeez, C++
  pub row_idx: Vec<i32>, // TODO: Is this right? Jeez, C++
  pub col_idx: Vec<i8>, // TODO: Is this right? Jeez, C++
  pub n_groups: i32,
  pub n_rows: i32,
  pub n_cols: i32,
}

#[derive(Copy,Clone,Debug)]
pub enum LayerType {
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

#[derive(Clone,Debug)]
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
  pub weight: Vec<Vec<f32>>, // TODO: Use actual matrices.
  pub bias: Vec<f32>,
  pub has_bias: bool,
  pub in_channels: i32,
  pub out_channels: i32,
  pub n_kernel: i32,
}

#[derive(Clone,Debug)]
pub struct Conv2dLayer {
  /* class Conv2dLayer : public TorchLayer{
    struct  Header{
        int elSize;  //size of each entry in bytes: 4 for float, 2 for fp16.
        int nKernel;  //kernel size. special case of conv2d used in WaveRNN
    };
    Vectorf weight;
    int nKernel;
  */
  pub weight: Vec<f32>,
  pub n_kernel: i32,
}

#[derive(Clone,Debug)]
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

  pub weight: Vec<f32>,
  pub bias: Vec<f32>,
  pub running_mean: Vec<f32>,
  pub running_var: Vec<f32>,
  pub eps: f32,
  pub n_channels: i32,
}

#[derive(Clone,Debug)]
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

  // TODO: CompMatrix mat;

  pub bias: Vec<f32>,
  pub n_rows: i32,
  pub n_cols: i32,
}

#[derive(Clone,Debug)]
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

  pub b_ir: Vec<f32>,
  pub b_iz: Vec<f32>,
  pub b_in: Vec<f32>,

  pub b_hr: Vec<f32>,
  pub b_hz: Vec<f32>,
  pub b_hn: Vec<f32>,

  pub n_rows: i32,
  pub n_cols: i32,
}

#[derive(Clone,Debug)]
pub struct Stretch2dLayer {
  /* class Stretch2dLayer : public TorchLayer{
    struct  Header{
        int x_scale;
        int y_scale;
    };
    int x_scale;
    int y_scale;
  */

  pub x_scale: i32,
  pub y_scale: i32,
}

fn read_into_matrix(file: &mut File, mat: &mut Vec<f32>) -> io::Result<()> {
  for (j, element) in enumerate(mat) {
    *element = file.read_f32::<LittleEndian>().expect("This should work");
  }
  Ok(())
}