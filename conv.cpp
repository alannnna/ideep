#include <ideep.hpp>
#include <ideep_pin_singletons.hpp>
#include <iostream>

#include "mkldnn_types.h" // mkldnn_status_t

using namespace ideep;

struct test_convolution_sizes_t {
    test_convolution_sizes_t(
        int mb,
        int ng,
        int ic, int ih, int iw,
        int oc, int oh, int ow,
        int kh, int kw,
        int padh, int padw,
        int strh, int strw,
        int dilh=0, int dilw=0
    ) :
        mb(mb),
        ng(ng),
        ic(ic), ih(ih), iw(iw),
        oc(oc), oh(oh), ow(ow),
        kh(kh), kw(kw),
        padh(padh), padw(padw),
        strh(strh), strw(strw),
        dilh(dilh), dilw(dilw) {}
    int mb;
    int ng;
    int ic, ih, iw;
    int oc, oh, ow;
    int kh, kw;
    int padh, padw;
    int strh, strw;
    int dilh, dilw;
};

static inline tensor make_output(void *buf = nullptr) {
  tensor ret;
  ret.set_data_handle(buf);
  return ret;
}

// template <typename data_t>
// static inline data_t set_value(size_t index, data_t mean, data_t deviation,
//         double sparsity)
// {
//     if (data_traits<data_t>::data_type == mkldnn::memory::data_type::f32) {
//         const size_t group_size = (size_t)(1. / sparsity);
//         const size_t group = index / group_size;
//         const size_t in_group = index % group_size;
//         const bool fill = in_group == ((group % 1637) % group_size);
//         return fill ? static_cast<data_t>(mean + deviation * sinf(float(index % 37)))
//             : data_t{0};
//     } else if (data_traits<data_t>::data_type == mkldnn::memory::data_type::s32
//         || data_traits<data_t>::data_type == mkldnn::memory::data_type::s16
//         || data_traits<data_t>::data_type == mkldnn::memory::data_type::s8) {
//         return data_t(rand() % 21 - 10);
//     } else if (data_traits<data_t>::data_type == mkldnn::memory::data_type::u8) {
//         return data_t(rand() % 17);
//     } else {
//         return data_t(0);
//     }
// }

// template <typename data_t>
// static void fill_data(const size_t size, data_t *data, double sparsity = 1.,
//         bool init_negs = false)
// {
//     for (ptrdiff_t n = 0; n < (ptrdiff_t)size; n++) {
//         data[n] = set_value<data_t>(n, data_t(1), data_t(2e-1), sparsity);

//         if (init_negs && n%4 == 0U)
//             data[n] = static_cast<data_t>(-data[n]); // weird for unsigned types!
//     }
// }

int main() {
  test_convolution_sizes_t cd{
    128, // int mb,
    1, // int ng,
    3, 32, 32, // int ic, int ih, int iw,
    64, 32, 32, // int oc, int oh, int ow,
    3, 3, // int kh, int kw,
    1, 1, // int padh, int padw,
    1, 1 // int strh, int strw,
  };

  tensor src_, weights_, bias_;
  tensor::dims dst_dims_;
  tensor::dims padR_;
  bool with_bias_;

    tensor::descriptor src_desc ({cd.mb, cd.ic, cd.ih, cd.iw},
        memory::data_type::f32,
        format_tag::nchw);//static_cast<format>(mkldnn::memory::format::nchw));

    auto weights_desc = tensor::descriptor(
          {cd.oc, cd.ic, cd.kh, cd.kw},
          memory::data_type::f32,
          format_tag::oihw);

    with_bias_ = false; // p.formats.bias_format != static_cast<mkldnn_memory_format_t>(format::format_undef);
    auto bias_desc = // with_bias_ ?
        //   tensor::descriptor({cd.oc}, data_traits<data_t_dst>::data_type,
        //       static_cast<format>(p.formats.bias_format)) :
            tensor::descriptor({}, memory::data_type::f32,
              format_tag::undef);

    src_.init(src_desc);
    weights_.init(weights_desc);
    bias_.init(bias_desc);

    // TODO: not sure if this needs to be fixed. Helpers commented out above.
    // fill_data<float>(
    //     src_.get_size() / sizeof(data_t_src),
    //     reinterpret_cast<data_t_src *>(src_.get_data_handle()));
    // fill_data<float>(
    //     weights_.get_size() / sizeof(data_t_src),
    //     reinterpret_cast<data_t_src *>(weights_.get_data_handle()));

    // if (with_bias_) {
    //   fill_data<float>(
    //       bias_.get_size() / sizeof(data_t_dst),
    //       reinterpret_cast<data_t_src *>(bias_.get_data_handle()));
    // }

    padR_ =  {cd.padh, cd.padw};
    for (int i = 0; i < 2; ++ i) {
      if ((cd.ih - ((cd.kh - 1) * (cd.dilh + 1) + 1) + cd.padh + padR_[0])
        / cd.strh + 1 != cd.oh)
        ++padR_[0];
      if ((cd.iw - ((cd.kw - 1) * (cd.dilw + 1) + 1) + cd.padw + padR_[1])
        / cd.strw + 1 != cd.ow)
        ++padR_[1];
    }

    dst_dims_ = {cd.mb, cd.oc, cd.oh, cd.ow};

  auto dst = make_output();

  // TestCommon(); // TODO

  /*
    // 2-in-1 compute (prepare & compute) without bias
  template <bool plain_format = false>
  static void compute(const tensor& src,
                      const tensor& weights,
                      const dims& dst_dims,
                      tensor& dst,
                      const dims& strides,
                      const dims& dilates,
                      const dims& padding_l,
                      const dims& padding_r,
                      int groups,
                      const scale_t& src_scales = scale_t(),
                      const scale_t& weights_scales = scale_t(),
                      const scale_t& dst_scales = scale_t(),
                      const attr_t& attr = attr_t(),
                      algorithm aalgorithm = algorithm::convolution_direct,
                      prop_kind aprop_kind = prop_kind::forward,
                      const lowp_kind alowp_kind = u8s8,
                      const engine& aengine = engine::cpu_engine())
  */
  for (size_t i = 0; i < 100000; i++) {
    convolution_forward::compute(src_, weights_, dst_dims_, dst,
        tensor::dims {cd.strh, cd.strw }, // strides
        tensor::dims {cd.dilh, cd.dilw }, // dilates
        tensor::dims {cd.padh, cd.padw }, // padding_l
        padR_, // padding_r
        1 // groups
    );
  }

// Check value
//   tensor ref_dst(dst.get_descriptor());
//   test_convolution_attr_t attr = p.attr;
//   attr.mkldnn_attr_recreate();
//   compute_ref_conv_fwd<float, float, float, float>(
//       cd, attr, src_, weights_, bias_, ref_dst);

//   compare_tensor<float>(ref_dst, dst);

    std::cout << "hi" << std::endl;
}
