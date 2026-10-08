```
pto::Tile<
  pto::TileType Loc_,
  Element_,
  Rows_,
  Cols_,
  pto::BLayout BLayout_      = pto::BLayout::RowMajor,
  RowValid_                  = Rows_,
  ColValid_                  = Cols_,
  pto::SLayout SLayout_      = pto::SLayout::NoneBox,
  SFractalSize_              = pto::TileConfig::fractalABSize,
  pto::PadValue PadValue_    = pto::PadValue::Null
>
```


a Tile is to describe the actual matrix elements to compute
[pto-isa/docs/coding/Tile.md at main · hw-native-sys/pto-isa](https://github.com/hw-native-sys/pto-isa/blob/main/docs/coding/Tile.md)

```
template <typename Element_, typename Shape_, typename Stride_, pto::Layout Layout_ = pto::Layout::ND>
struct GlobalTensor;
```

a GlobalTensor is to describe the memory layout and memory access pattern of a tensor
[pto-isa/docs/coding/GlobalTensor.md at main · hw-native-sys/pto-isa](https://github.com/hw-native-sys/pto-isa/blob/main/docs/coding/GlobalTensor.md)

## Basic:
memory stuff:

[[row major&col major]]
[[matrix stride(mapping)]]


