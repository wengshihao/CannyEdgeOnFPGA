/*
The canny_edge IP is based on the work of Aadeetya Shreedhar and Alexander Wang
https://github.com/ka367/Lane-Detecting-Using-Hough-Transform/blob/master/test.cpp
*/
#include "canny_edge.h"
#include<assert.h>
void gradient_decomposition(GRAY_IMAGE_16S& gx, GRAY_IMAGE_16S& gy, GRAY_IMAGE_16& gd) {

	HLS_SIZE_T rows = gx.rows;
	HLS_SIZE_T cols = gx.cols;

	hls::Scalar<1, short> pixel_gx;
	hls::Scalar<1, short> pixel_gy;
	hls::Scalar<1, unsigned short> element_pixel;
	hls::Scalar<1, unsigned char> dir_pixel;

	short element_gx;
	short element_gy;
	unsigned short abs_gx;
	unsigned short abs_gy;
	unsigned short abs_g;
	unsigned short dir_g;
	unsigned short element_gd;

	gradient_decomposition_label0:for( HLS_SIZE_T i = 0; i < rows; i++ ) {
		gradient_decomposition_label1:for( HLS_SIZE_T j = 0; j < cols; j++ ) {
#pragma HLS PIPELINE enable_flush rewind
#pragma HLS LOOP_FLATTEN OFF
#pragma HLS DEPENDENCE array inter false

			gx >> pixel_gx;
			gy >> pixel_gy;
			element_gx = pixel_gx.val[0];
			element_gy = pixel_gy.val[0];

			abs_gx = hls::abs(element_gx);
			abs_gy = hls::abs(element_gy);

			abs_g  = abs_gx + abs_gy;
			if (abs_gx > abs_gy && ((element_gx > 0 && element_gy >= 0)||(element_gx < 0 && element_gy <= 0))) {
			  if (5*abs_gx > (12*abs_gy)) dir_g = 0;
			  else dir_g = 1;
			}
			else if (abs_gx <= abs_gy && ((element_gx > 0 && element_gy > 0)||(element_gx < 0 && element_gy < 0))) {
			  if (5*abs_gy > (12*abs_gx)) dir_g = 2;
			  else dir_g = 1;
			}
			else if (abs_gx <= abs_gy && ((element_gx >= 0 && element_gy < 0)||(element_gx <= 0 && element_gy > 0))) {
			  if (5*abs_gy > (12*abs_gx)) dir_g = 2;
			  else dir_g = 3;
			}
			else {
			  if(abs_gx== 0 && abs_gy == 0) dir_g = 0;
			  else if (5*abs_gx > (12*abs_gy)) dir_g = 0;
			  else dir_g = 3;
			}
			element_gd = ( ( abs_g << 2 ) | dir_g );
			element_pixel.val[0] = element_gd;
			gd << element_pixel;
		}
	}
}

void nonmax_suppression(GRAY_IMAGE_16& gd, GRAY_IMAGE_16& dst) {

	HLS_SIZE_T rows = gd.rows;
	HLS_SIZE_T cols = gd.cols;

	hls::LineBuffer<2, 1920, unsigned short> linebuff;
	hls::Window<3, 3, unsigned short> win;
	hls::Scalar<1, unsigned short> pixel_gd;
	hls::Scalar<1, unsigned short> out_pixel;

	unsigned short element_gd;
	unsigned short out_pixel_val;
	unsigned char current_dir;
	unsigned short current_grad;
	unsigned short ga;
	unsigned short gb;
	unsigned short tmp0;
	unsigned short tmp1;
	nonmax_suppression_label2:for( HLS_SIZE_T i = 0; i < rows+1; i++ ) {
		nonmax_suppression_label3:for( HLS_SIZE_T j = 0; j < cols+1; j++ ) {
#pragma HLS LOOP_FLATTEN OFF
#pragma HLS DEPENDENCE array inter false


		  if ( i < rows && j < cols ) {
			gd >> pixel_gd;
			element_gd = pixel_gd.val[0];
		  }

		  if( j < cols ) {
			tmp1 = linebuff.getval(1, j);
			tmp0 = linebuff.getval(0, j);
			linebuff.val[1][j] = tmp0;
		  }
		  if( j < cols && i < rows ){
			linebuff.insert_bottom( element_gd, j );
		  }

		  win.shift_right();

		  if( j < cols ) {
			win.insert( element_gd, 0, 0 );
			win.insert( tmp0, 1, 0 );
			win.insert( tmp1, 2, 0 );
		  }

		  current_dir = win.getval(1, 1) & 3;
		  current_grad = win.getval(1, 1) >> 2;

		  if( i <= 1 || j <= 1 || i > rows-1 || j > cols-1 ) {
			out_pixel_val = 0;
		  }
		  else {
			if ( current_dir == 0/*0*/ ) {
			  ga = win.getval( 1, 0 )>>2;
			  gb = win.getval( 1, 2 )>>2;
			}
			else if ( current_dir == 3/*1*/ ){
			  ga = win.getval( 2, 0 )>>2;
			  gb = win.getval( 0, 2 )>>2;
			}
			else if ( current_dir == 2/*2*/ ){
			  ga = win.getval( 0, 1 )>>2;
			  gb = win.getval( 2, 1 )>>2;
			}
			else {
			  ga = win.getval( 2, 2 )>>2;
			  gb = win.getval( 0, 0 )>>2;
			}

			if( current_grad > ga && current_grad > gb ) {
			  out_pixel_val = current_grad;
			}
			else {
			  out_pixel_val = 0;
			}
		  }

		  if( j > 0 && i > 0 ) {
			out_pixel.val[0] = out_pixel_val;
			dst << out_pixel;
		  }

		}
	}
}

void hysteresis( GRAY_IMAGE_16& src, GRAY_IMAGE& dst, int threshold_low, int threshold_high ) {

	HLS_SIZE_T rows = src.rows;
	HLS_SIZE_T cols = src.cols;

	hls::LineBuffer<2, 1920, unsigned short> linebuff;
	hls::Window<3, 3, unsigned short> win;

	hls::Scalar<1, unsigned short> pixel_gd;
	hls::Scalar<1, unsigned char> out_pixel;

	unsigned short element_gd;
	unsigned char out_pixel_val;
	unsigned short tmp0;
	unsigned short tmp1;

	hysteresis_label4:for( HLS_SIZE_T i = 0; i < rows+1; i++ ) {
		hysteresis_label5:for( HLS_SIZE_T j = 0; j < cols+1; j++ ) {
#pragma HLS PIPELINE enable_flush rewind

#pragma HLS LOOP_FLATTEN OFF
#pragma HLS DEPENDENCE array inter false


		  if ( i < rows && j < cols ) {
		    src >> pixel_gd;
			element_gd = pixel_gd.val[0];
		  }

		  if( j < cols ) {
			tmp1 = linebuff.getval(1, j);
			tmp0 = linebuff.getval(0, j);
			linebuff.val[1][j] = tmp0;
		  }
		  if( j < cols && i < rows ){
			linebuff.insert_bottom( element_gd, j );
		  }

		  win.shift_right();

		  if( j < cols ) {
			win.insert( element_gd, 0, 0 );
			win.insert( tmp0, 1, 0 );
			win.insert( tmp1, 2, 0 );
		  }

		  if( i <= 1 || j <= 1 || i > rows-1 || j > cols-1 ) {
			out_pixel_val = 0;
		  }
		  else {
			if( win.getval(1,1) < threshold_low ){
			  out_pixel_val = 0;
			}
			else if( 	win.getval(1,1) > threshold_high ||
						win.getval(0,0) > threshold_high  ||
						win.getval(0,1) > threshold_high  ||
						win.getval(0,2) > threshold_high  ||
						win.getval(1,0) > threshold_high  ||
						win.getval(1,2) > threshold_high  ||
						win.getval(2,0) > threshold_high  ||
						win.getval(2,1) > threshold_high  ||
						win.getval(2,2) > threshold_high  ) {
			  out_pixel_val = 255;
			}
			else {
			  out_pixel_val = 0;
			}
		  }

		  if( j > 0 && i > 0 ) {
			out_pixel.val[0] = out_pixel_val;
			dst << out_pixel;
		  }
		}
	}
}

void canny_edge(wide_stream* in_stream1, wide_stream* out_stream1,
		wide_stream* in_stream2, wide_stream* out_stream2,
		wide_stream* in_stream3, wide_stream* out_stream3,
		wide_stream* in_stream4, wide_stream* out_stream4,
		ap_uint<32> rows, ap_uint<32> cols, int threshold1, int threshold2){

#pragma HLS INTERFACE axis port=in_stream1 bundle=INPUT1
#pragma HLS INTERFACE axis port=out_stream1 bundle=OUTPUT1
#pragma HLS INTERFACE axis port=in_stream2 bundle=INPUT2
#pragma HLS INTERFACE axis port=out_stream2 bundle=OUTPUT2
#pragma HLS INTERFACE axis port=in_stream3 bundle=INPUT3
#pragma HLS INTERFACE axis port=out_stream3 bundle=OUTPUT3
#pragma HLS INTERFACE axis port=in_stream4 bundle=INPUT4
#pragma HLS INTERFACE axis port=out_stream4 bundle=OUTPUT4

#pragma HLS INTERFACE s_axilite port=rows bundle=CONTROL_BUS offset=0x14 clock=AXI_LITE_clk
#pragma HLS INTERFACE s_axilite port=cols bundle=CONTROL_BUS offset=0x1C clock=AXI_LITE_clk
#pragma HLS INTERFACE s_axilite port=threshold1 bundle=CONTROL_BUS offset=0x24 clock=AXI_LITE_clk
#pragma HLS INTERFACE s_axilite port=threshold2 bundle=CONTROL_BUS offset=0x2C clock=AXI_LITE_clk
#pragma HLS INTERFACE s_axilite port=return bundle=CONTROL_BUS clock=AXI_LITE_clk

#pragma HLS INTERFACE ap_stable port=rows
#pragma HLS INTERFACE ap_stable port=cols
#pragma HLS INTERFACE ap_stable port=threshold1
#pragma HLS INTERFACE ap_stable port=threshold2

	GRAY_IMAGE src_bw1(rows, cols);
	GRAY_IMAGE src_bw2(rows, cols);
	GRAY_IMAGE src_bw3(rows, cols);
	GRAY_IMAGE src_bw4(rows, cols);

	GRAY_IMAGE src_blur1(rows, cols);
	GRAY_IMAGE src_blur2(rows, cols);
	GRAY_IMAGE src_blur3(rows, cols);
	GRAY_IMAGE src_blur4(rows, cols);

	GRAY_IMAGE src11(rows, cols);
	GRAY_IMAGE src12(rows, cols);
	GRAY_IMAGE src21(rows, cols);
	GRAY_IMAGE src22(rows, cols);
	GRAY_IMAGE src31(rows, cols);
	GRAY_IMAGE src32(rows, cols);
	GRAY_IMAGE src41(rows, cols);
	GRAY_IMAGE src42(rows, cols);

	GRAY_IMAGE_16S sobel_gx1(rows, cols);
	GRAY_IMAGE_16S sobel_gy1(rows, cols);
	GRAY_IMAGE_16S sobel_gx2(rows, cols);
	GRAY_IMAGE_16S sobel_gy2(rows, cols);
	GRAY_IMAGE_16S sobel_gx3(rows, cols);
	GRAY_IMAGE_16S sobel_gy3(rows, cols);
	GRAY_IMAGE_16S sobel_gx4(rows, cols);
	GRAY_IMAGE_16S sobel_gy4(rows, cols);

    GRAY_IMAGE_16 grad_gd1(rows, cols);
    GRAY_IMAGE_16 grad_gd2(rows, cols);
    GRAY_IMAGE_16 grad_gd3(rows, cols);
    GRAY_IMAGE_16 grad_gd4(rows, cols);

    GRAY_IMAGE_16 suppressed1(rows, cols);
    GRAY_IMAGE_16 suppressed2(rows, cols);
    GRAY_IMAGE_16 suppressed3(rows, cols);
    GRAY_IMAGE_16 suppressed4(rows, cols);

    GRAY_IMAGE thresholded1(rows, cols);
    GRAY_IMAGE thresholded2(rows, cols);
    GRAY_IMAGE thresholded3(rows, cols);
    GRAY_IMAGE thresholded4(rows, cols);

    GRAY_IMAGE canny_edges1(rows, cols);
    GRAY_IMAGE canny_edges2(rows, cols);
    GRAY_IMAGE canny_edges3(rows, cols);
    GRAY_IMAGE canny_edges4(rows, cols);

    #pragma HLS dataflow
	const int col_packets = cols/4;
	const int packets = col_packets*rows;
	const int pixel_cnt = rows*cols;

	for(int r = 0; r < packets; r++){
#pragma HLS PIPELINE enable_flush rewind

		ap_uint<32> dat1 = in_stream1->data;
		src_bw1.write(GRAY_PIXEL(dat1.range(7,0)));
		src_bw1.write(GRAY_PIXEL(dat1.range(15,8)));
		src_bw1.write(GRAY_PIXEL(dat1.range(23,16)));
		src_bw1.write(GRAY_PIXEL(dat1.range(31,24)));
		++in_stream1;

		ap_uint<32> dat2 = in_stream2->data;
		src_bw2.write(GRAY_PIXEL(dat2.range(7,0)));
		src_bw2.write(GRAY_PIXEL(dat2.range(15,8)));
		src_bw2.write(GRAY_PIXEL(dat2.range(23,16)));
		src_bw2.write(GRAY_PIXEL(dat2.range(31,24)));
		++in_stream2;

		ap_uint<32> dat3 = in_stream3->data;
		src_bw3.write(GRAY_PIXEL(dat3.range(7,0)));
		src_bw3.write(GRAY_PIXEL(dat3.range(15,8)));
		src_bw3.write(GRAY_PIXEL(dat3.range(23,16)));
		src_bw3.write(GRAY_PIXEL(dat3.range(31,24)));
		++in_stream3;

		ap_uint<32> dat4 = in_stream4->data;
		src_bw4.write(GRAY_PIXEL(dat4.range(7,0)));
		src_bw4.write(GRAY_PIXEL(dat4.range(15,8)));
		src_bw4.write(GRAY_PIXEL(dat4.range(23,16)));
		src_bw4.write(GRAY_PIXEL(dat4.range(31,24)));
		++in_stream4;
	}

	hls::Duplicate( src_bw1, src11, src12 );
	hls::Duplicate( src_bw2, src21, src22 );
	hls::Duplicate( src_bw3, src31, src32 );
	hls::Duplicate( src_bw4, src41, src42 );

    hls::Sobel<1,0,3>( src11, sobel_gx1 );
    hls::Sobel<0,1,3>( src12, sobel_gy1 );
    hls::Sobel<1,0,3>( src21, sobel_gx2 );
	hls::Sobel<0,1,3>( src22, sobel_gy2 );
	hls::Sobel<1,0,3>( src31, sobel_gx3 );
	hls::Sobel<0,1,3>( src32, sobel_gy3 );
	hls::Sobel<1,0,3>( src41, sobel_gx4 );
	hls::Sobel<0,1,3>( src42, sobel_gy4 );

	gradient_decomposition( sobel_gx1, sobel_gy1, grad_gd1 );
	gradient_decomposition( sobel_gx2, sobel_gy2, grad_gd2 );
	gradient_decomposition( sobel_gx3, sobel_gy3, grad_gd3 );
	gradient_decomposition( sobel_gx4, sobel_gy4, grad_gd4 );

	nonmax_suppression( grad_gd1, suppressed1 );
	nonmax_suppression( grad_gd2, suppressed2 );
	nonmax_suppression( grad_gd3, suppressed3 );
	nonmax_suppression( grad_gd4, suppressed4 );

	hysteresis( suppressed1, canny_edges1, threshold1, threshold2);
	hysteresis( suppressed2, canny_edges2, threshold1, threshold2);
	hysteresis( suppressed3, canny_edges3, threshold1, threshold2);
	hysteresis( suppressed4, canny_edges4, threshold1, threshold2);

    for(int r = 0; r < rows; r++){
#pragma HLS PIPELINE enable_flush rewind
		for(int c = 0; c < col_packets; c++){

			ap_uint<32> dat1;
			dat1.range(7,0) = canny_edges1.read().val[0];
			dat1.range(15,8) = canny_edges1.read().val[0];
			dat1.range(23,16) = canny_edges1.read().val[0];
			dat1.range(31,24) = canny_edges1.read().val[0];
			out_stream1->data = dat1;
			out_stream1->user = (r == 0 && c == 0)? 1: 0;
			out_stream1->last = (r == rows-1 && c == col_packets-1)? 1: 0;
			++out_stream1;

			ap_uint<32> dat2;
			dat2.range(7,0) = canny_edges2.read().val[0];
			dat2.range(15,8) = canny_edges2.read().val[0];
			dat2.range(23,16) = canny_edges2.read().val[0];
			dat2.range(31,24) = canny_edges2.read().val[0];
			out_stream2->data = dat2;
			out_stream2->user = (r == 0 && c == 0)? 1: 0;
			out_stream2->last = (r == rows-1 && c == col_packets-1)? 1: 0;
			++out_stream2;

			ap_uint<32> dat3;
			dat3.range(7,0) = canny_edges3.read().val[0];
			dat3.range(15,8) = canny_edges3.read().val[0];
			dat3.range(23,16) = canny_edges3.read().val[0];
			dat3.range(31,24) = canny_edges3.read().val[0];
			out_stream3->data = dat3;
			out_stream3->user = (r == 0 && c == 0)? 1: 0;
			out_stream3->last = (r == rows-1 && c == col_packets-1)? 1: 0;
			++out_stream3;

			ap_uint<32> dat4;
			dat4.range(7,0) = canny_edges4.read().val[0];
			dat4.range(15,8) = canny_edges4.read().val[0];
			dat4.range(23,16) = canny_edges4.read().val[0];
			dat4.range(31,24) = canny_edges4.read().val[0];
			out_stream4->data = dat4;
			out_stream4->user = (r == 0 && c == 0)? 1: 0;
			out_stream4->last = (r == rows-1 && c == col_packets-1)? 1: 0;
			++out_stream4;


		}
	}
}

