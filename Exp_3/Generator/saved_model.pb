??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResizeNearestNeighbor
images"T
size
resized_images"T"
Ttype:
2
	"
align_cornersbool( "
half_pixel_centersbool( 
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
-
Tanh
x"T
y"T"
Ttype:

2
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??	
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??b*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
??b*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?b*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:?b*
dtype0
?
conv2d_6/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_6/kernel
}
#conv2d_6/kernel/Read/ReadVariableOpReadVariableOpconv2d_6/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_6/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_6/bias
l
!conv2d_6/bias/Read/ReadVariableOpReadVariableOpconv2d_6/bias*
_output_shapes	
:?*
dtype0
?
conv2d_7/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_7/kernel
{
#conv2d_7/kernel/Read/ReadVariableOpReadVariableOpconv2d_7/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_7/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_7/bias
k
!conv2d_7/bias/Read/ReadVariableOpReadVariableOpconv2d_7/bias*
_output_shapes
:@*
dtype0
?
conv2d_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_8/kernel
{
#conv2d_8/kernel/Read/ReadVariableOpReadVariableOpconv2d_8/kernel*&
_output_shapes
: *
dtype0
r
conv2d_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_8/bias
k
!conv2d_8/bias/Read/ReadVariableOpReadVariableOpconv2d_8/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
	dense

activation
reshape
upsample
	conv1
	conv2
	conv3
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
h

%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
h

+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
8
0
1
2
 3
%4
&5
+6
,7
 
8
0
1
2
 3
%4
&5
+6
,7
?

1layers
trainable_variables
2non_trainable_variables
	regularization_losses
3metrics
4layer_regularization_losses
5layer_metrics

	variables
 
KI
VARIABLE_VALUEdense_1/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEdense_1/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

6layers
trainable_variables
7non_trainable_variables
regularization_losses
8metrics
9layer_regularization_losses
:layer_metrics
	variables
 
 
 
?

;layers
trainable_variables
<non_trainable_variables
regularization_losses
=metrics
>layer_regularization_losses
?layer_metrics
	variables
 
 
 
?

@layers
trainable_variables
Anon_trainable_variables
regularization_losses
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
 
 
 
?

Elayers
trainable_variables
Fnon_trainable_variables
regularization_losses
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
LJ
VARIABLE_VALUEconv2d_6/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_6/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
 1
 

0
 1
?

Jlayers
!trainable_variables
Knon_trainable_variables
"regularization_losses
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
#	variables
LJ
VARIABLE_VALUEconv2d_7/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_7/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

%0
&1
 

%0
&1
?

Olayers
'trainable_variables
Pnon_trainable_variables
(regularization_losses
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
)	variables
LJ
VARIABLE_VALUEconv2d_8/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_8/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

+0
,1
 

+0
,1
?

Tlayers
-trainable_variables
Unon_trainable_variables
.regularization_losses
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
/	variables
1
0
1
2
3
4
5
6
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_1/kerneldense_1/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? */
f*R(
&__inference_signature_wrapper_28087274
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp#conv2d_6/kernel/Read/ReadVariableOp!conv2d_6/bias/Read/ReadVariableOp#conv2d_7/kernel/Read/ReadVariableOp!conv2d_7/bias/Read/ReadVariableOp#conv2d_8/kernel/Read/ReadVariableOp!conv2d_8/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? **
f%R#
!__inference__traced_save_28087923
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1/kerneldense_1/biasconv2d_6/kernelconv2d_6/biasconv2d_7/kernelconv2d_7/biasconv2d_8/kernelconv2d_8/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference__traced_restore_28087957??	
?h
?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087351

inputs:
&dense_1_matmul_readvariableop_resource:
??b6
'dense_1_biasadd_readvariableop_resource:	?bC
'conv2d_6_conv2d_readvariableop_resource:??7
(conv2d_6_biasadd_readvariableop_resource:	?A
'conv2d_7_conv2d_readvariableop_resource:@@6
(conv2d_7_biasadd_readvariableop_resource:@A
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource:
identity??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_1/BiasAdd?
glu_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_1/strided_slice/stack?
glu_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_1/strided_slice/stack_1?
glu_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_1/strided_slice/stack_2?
glu_1/strided_sliceStridedSlicedense_1/BiasAdd:output:0"glu_1/strided_slice/stack:output:0$glu_1/strided_slice/stack_1:output:0$glu_1/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_1/strided_slice?
glu_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_1/strided_slice_1/stack?
glu_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_1/strided_slice_1/stack_1?
glu_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_1/strided_slice_1/stack_2?
glu_1/strided_slice_1StridedSlicedense_1/BiasAdd:output:0$glu_1/strided_slice_1/stack:output:0&glu_1/strided_slice_1/stack_1:output:0&glu_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_1/strided_slice_1|
glu_1/SigmoidSigmoidglu_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_1/Sigmoid?
	glu_1/mulMulglu_1/strided_slice:output:0glu_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_1/mul_
reshape_1/ShapeShapeglu_1/mul:z:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeglu_1/mul:z:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshape
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborreshape_1/Reshape:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
glu_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_2/stack?
glu_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_1/strided_slice_2/stack_1?
glu_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_2/stack_2?
glu_1/strided_slice_2StridedSliceconv2d_6/BiasAdd:output:0$glu_1/strided_slice_2/stack:output:0&glu_1/strided_slice_2/stack_1:output:0&glu_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_1/strided_slice_2?
glu_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_1/strided_slice_3/stack?
glu_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_3/stack_1?
glu_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_3/stack_2?
glu_1/strided_slice_3StridedSliceconv2d_6/BiasAdd:output:0$glu_1/strided_slice_3/stack:output:0&glu_1/strided_slice_3/stack_1:output:0&glu_1/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_1/strided_slice_3?
glu_1/Sigmoid_1Sigmoidglu_1/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_1/Sigmoid_1?
glu_1/mul_1Mulglu_1/strided_slice_2:output:0glu_1/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_1/mul_1?
up_sampling2d_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_2?
up_sampling2d_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_3?
up_sampling2d_1/mul_1Mul up_sampling2d_1/Const_2:output:0 up_sampling2d_1/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul_1?
.up_sampling2d_1/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_1/mul_1:z:0up_sampling2d_1/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_1/resize_1/ResizeNearestNeighbor?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D?up_sampling2d_1/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_7/BiasAdd?
glu_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_4/stack?
glu_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_4/stack_1?
glu_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_4/stack_2?
glu_1/strided_slice_4StridedSliceconv2d_7/BiasAdd:output:0$glu_1/strided_slice_4/stack:output:0&glu_1/strided_slice_4/stack_1:output:0&glu_1/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_1/strided_slice_4?
glu_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_5/stack?
glu_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_5/stack_1?
glu_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_5/stack_2?
glu_1/strided_slice_5StridedSliceconv2d_7/BiasAdd:output:0$glu_1/strided_slice_5/stack:output:0&glu_1/strided_slice_5/stack_1:output:0&glu_1/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_1/strided_slice_5?
glu_1/Sigmoid_2Sigmoidglu_1/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_1/Sigmoid_2?
glu_1/mul_2Mulglu_1/strided_slice_4:output:0glu_1/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_1/mul_2?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dglu_1/mul_2:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_8/BiasAdd{
conv2d_8/TanhTanhconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_8/Tanh?
IdentityIdentityconv2d_8/Tanh:y:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
.__inference_generator_1_layer_call_fn_28087666
input_1
unknown:
??b
	unknown_0:	?b%
	unknown_1:??
	unknown_2:	?#
	unknown_3:@@
	unknown_4:@#
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_generator_1_layer_call_and_return_conditional_losses_280871512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
`
C__inference_glu_1_layer_call_and_return_conditional_losses_28086830
input_1
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice_1j
SigmoidSigmoidstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????12	
Sigmoidi
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:Q M
(
_output_shapes
:??????????b
!
_user_specified_name	input_1
?
`
C__inference_glu_1_layer_call_and_return_conditional_losses_28086844
input_1
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinput_1strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinput_1strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice_1j
SigmoidSigmoidstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????12	
Sigmoidi
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:Q M
(
_output_shapes
:??????????b
!
_user_specified_name	input_1
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28087741

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+??????????????????????????? *

begin_mask*
end_mask2
strided_slice?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+??????????????????????????? *

begin_mask*
end_mask2
strided_slice_1?
SigmoidSigmoidstrided_slice_1:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
Sigmoid?
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
D
(__inference_glu_1_layer_call_fn_28087799

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280870692
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?h
?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087428

inputs:
&dense_1_matmul_readvariableop_resource:
??b6
'dense_1_biasadd_readvariableop_resource:	?bC
'conv2d_6_conv2d_readvariableop_resource:??7
(conv2d_6_biasadd_readvariableop_resource:	?A
'conv2d_7_conv2d_readvariableop_resource:@@6
(conv2d_7_biasadd_readvariableop_resource:@A
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource:
identity??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinputs%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_1/BiasAdd?
glu_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_1/strided_slice/stack?
glu_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_1/strided_slice/stack_1?
glu_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_1/strided_slice/stack_2?
glu_1/strided_sliceStridedSlicedense_1/BiasAdd:output:0"glu_1/strided_slice/stack:output:0$glu_1/strided_slice/stack_1:output:0$glu_1/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_1/strided_slice?
glu_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_1/strided_slice_1/stack?
glu_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_1/strided_slice_1/stack_1?
glu_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_1/strided_slice_1/stack_2?
glu_1/strided_slice_1StridedSlicedense_1/BiasAdd:output:0$glu_1/strided_slice_1/stack:output:0&glu_1/strided_slice_1/stack_1:output:0&glu_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_1/strided_slice_1|
glu_1/SigmoidSigmoidglu_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_1/Sigmoid?
	glu_1/mulMulglu_1/strided_slice:output:0glu_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_1/mul_
reshape_1/ShapeShapeglu_1/mul:z:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeglu_1/mul:z:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshape
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborreshape_1/Reshape:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
glu_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_2/stack?
glu_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_1/strided_slice_2/stack_1?
glu_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_2/stack_2?
glu_1/strided_slice_2StridedSliceconv2d_6/BiasAdd:output:0$glu_1/strided_slice_2/stack:output:0&glu_1/strided_slice_2/stack_1:output:0&glu_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_1/strided_slice_2?
glu_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_1/strided_slice_3/stack?
glu_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_3/stack_1?
glu_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_3/stack_2?
glu_1/strided_slice_3StridedSliceconv2d_6/BiasAdd:output:0$glu_1/strided_slice_3/stack:output:0&glu_1/strided_slice_3/stack_1:output:0&glu_1/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_1/strided_slice_3?
glu_1/Sigmoid_1Sigmoidglu_1/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_1/Sigmoid_1?
glu_1/mul_1Mulglu_1/strided_slice_2:output:0glu_1/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_1/mul_1?
up_sampling2d_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_2?
up_sampling2d_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_3?
up_sampling2d_1/mul_1Mul up_sampling2d_1/Const_2:output:0 up_sampling2d_1/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul_1?
.up_sampling2d_1/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_1/mul_1:z:0up_sampling2d_1/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_1/resize_1/ResizeNearestNeighbor?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D?up_sampling2d_1/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_7/BiasAdd?
glu_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_4/stack?
glu_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_4/stack_1?
glu_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_4/stack_2?
glu_1/strided_slice_4StridedSliceconv2d_7/BiasAdd:output:0$glu_1/strided_slice_4/stack:output:0&glu_1/strided_slice_4/stack_1:output:0&glu_1/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_1/strided_slice_4?
glu_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_5/stack?
glu_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_5/stack_1?
glu_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_5/stack_2?
glu_1/strided_slice_5StridedSliceconv2d_7/BiasAdd:output:0$glu_1/strided_slice_5/stack:output:0&glu_1/strided_slice_5/stack_1:output:0&glu_1/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_1/strided_slice_5?
glu_1/Sigmoid_2Sigmoidglu_1/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_1/Sigmoid_2?
glu_1/mul_2Mulglu_1/strided_slice_4:output:0glu_1/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_1/mul_2?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dglu_1/mul_2:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_8/BiasAdd{
conv2d_8/TanhTanhconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_8/Tanh?
IdentityIdentityconv2d_8/Tanh:y:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
.__inference_generator_1_layer_call_fn_28087624

inputs
unknown:
??b
	unknown_0:	?b%
	unknown_1:??
	unknown_2:	?#
	unknown_3:@@
	unknown_4:@#
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_generator_1_layer_call_and_return_conditional_losses_280869852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_6_layer_call_fn_28087837

inputs#
unknown:??
	unknown_0:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_280869142
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?+
?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087151

inputs$
dense_1_28087124:
??b
dense_1_28087126:	?b-
conv2d_6_28087132:?? 
conv2d_6_28087134:	?+
conv2d_7_28087139:@@
conv2d_7_28087141:@+
conv2d_8_28087145: 
conv2d_8_28087147:
identity?? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_28087124dense_1_28087126*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_280868802!
dense_1/StatefulPartitionedCall?
glu_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280868082
glu_1/PartitionedCall?
reshape_1/PartitionedCallPartitionedCallglu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_280869012
reshape_1/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_280868572!
up_sampling2d_1/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_28087132conv2d_6_28087134*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_280869142"
 conv2d_6/StatefulPartitionedCall?
glu_1/PartitionedCall_1PartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280870692
glu_1/PartitionedCall_1?
!up_sampling2d_1/PartitionedCall_1PartitionedCall glu_1/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_280868572#
!up_sampling2d_1/PartitionedCall_1?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*up_sampling2d_1/PartitionedCall_1:output:0conv2d_7_28087139conv2d_7_28087141*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_280869462"
 conv2d_7/StatefulPartitionedCall?
glu_1/PartitionedCall_2PartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280870352
glu_1/PartitionedCall_2?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall glu_1/PartitionedCall_2:output:0conv2d_8_28087145conv2d_8_28087147*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_280869782"
 conv2d_8/StatefulPartitionedCall?
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_8_layer_call_and_return_conditional_losses_28086978

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
*__inference_dense_1_layer_call_fn_28087685

inputs
unknown:
??b
	unknown_0:	?b
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_280868802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28086784

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice_1j
SigmoidSigmoidstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????12	
Sigmoidi
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?h
?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087582
input_1:
&dense_1_matmul_readvariableop_resource:
??b6
'dense_1_biasadd_readvariableop_resource:	?bC
'conv2d_6_conv2d_readvariableop_resource:??7
(conv2d_6_biasadd_readvariableop_resource:	?A
'conv2d_7_conv2d_readvariableop_resource:@@6
(conv2d_7_biasadd_readvariableop_resource:@A
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource:
identity??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinput_1%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_1/BiasAdd?
glu_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_1/strided_slice/stack?
glu_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_1/strided_slice/stack_1?
glu_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_1/strided_slice/stack_2?
glu_1/strided_sliceStridedSlicedense_1/BiasAdd:output:0"glu_1/strided_slice/stack:output:0$glu_1/strided_slice/stack_1:output:0$glu_1/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_1/strided_slice?
glu_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_1/strided_slice_1/stack?
glu_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_1/strided_slice_1/stack_1?
glu_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_1/strided_slice_1/stack_2?
glu_1/strided_slice_1StridedSlicedense_1/BiasAdd:output:0$glu_1/strided_slice_1/stack:output:0&glu_1/strided_slice_1/stack_1:output:0&glu_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_1/strided_slice_1|
glu_1/SigmoidSigmoidglu_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_1/Sigmoid?
	glu_1/mulMulglu_1/strided_slice:output:0glu_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_1/mul_
reshape_1/ShapeShapeglu_1/mul:z:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeglu_1/mul:z:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshape
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborreshape_1/Reshape:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
glu_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_2/stack?
glu_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_1/strided_slice_2/stack_1?
glu_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_2/stack_2?
glu_1/strided_slice_2StridedSliceconv2d_6/BiasAdd:output:0$glu_1/strided_slice_2/stack:output:0&glu_1/strided_slice_2/stack_1:output:0&glu_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_1/strided_slice_2?
glu_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_1/strided_slice_3/stack?
glu_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_3/stack_1?
glu_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_3/stack_2?
glu_1/strided_slice_3StridedSliceconv2d_6/BiasAdd:output:0$glu_1/strided_slice_3/stack:output:0&glu_1/strided_slice_3/stack_1:output:0&glu_1/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_1/strided_slice_3?
glu_1/Sigmoid_1Sigmoidglu_1/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_1/Sigmoid_1?
glu_1/mul_1Mulglu_1/strided_slice_2:output:0glu_1/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_1/mul_1?
up_sampling2d_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_2?
up_sampling2d_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_3?
up_sampling2d_1/mul_1Mul up_sampling2d_1/Const_2:output:0 up_sampling2d_1/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul_1?
.up_sampling2d_1/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_1/mul_1:z:0up_sampling2d_1/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_1/resize_1/ResizeNearestNeighbor?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D?up_sampling2d_1/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_7/BiasAdd?
glu_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_4/stack?
glu_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_4/stack_1?
glu_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_4/stack_2?
glu_1/strided_slice_4StridedSliceconv2d_7/BiasAdd:output:0$glu_1/strided_slice_4/stack:output:0&glu_1/strided_slice_4/stack_1:output:0&glu_1/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_1/strided_slice_4?
glu_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_5/stack?
glu_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_5/stack_1?
glu_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_5/stack_2?
glu_1/strided_slice_5StridedSliceconv2d_7/BiasAdd:output:0$glu_1/strided_slice_5/stack:output:0&glu_1/strided_slice_5/stack_1:output:0&glu_1/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_1/strided_slice_5?
glu_1/Sigmoid_2Sigmoidglu_1/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_1/Sigmoid_2?
glu_1/mul_2Mulglu_1/strided_slice_4:output:0glu_1/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_1/mul_2?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dglu_1/mul_2:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_8/BiasAdd{
conv2d_8/TanhTanhconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_8/Tanh?
IdentityIdentityconv2d_8/Tanh:y:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
H
,__inference_reshape_1_layer_call_fn_28087818

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_280869012
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameinputs
?
?
F__inference_conv2d_7_layer_call_and_return_conditional_losses_28087847

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_28086901

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameinputs
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28087069

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+???????????????????????????@*

begin_mask*
end_mask2
strided_slice?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+???????????????????????????@*

begin_mask*
end_mask2
strided_slice_1?
SigmoidSigmoidstrided_slice_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
Sigmoid?
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?%
?
$__inference__traced_restore_28087957
file_prefix3
assignvariableop_dense_1_kernel:
??b.
assignvariableop_1_dense_1_bias:	?b>
"assignvariableop_2_conv2d_6_kernel:??/
 assignvariableop_3_conv2d_6_bias:	?<
"assignvariableop_4_conv2d_7_kernel:@@.
 assignvariableop_5_conv2d_7_bias:@<
"assignvariableop_6_conv2d_8_kernel: .
 assignvariableop_7_conv2d_8_bias:

identity_9??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_dense_1_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_1_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_6_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_6_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_7_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_7_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_8_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_8_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8?

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*%
_input_shapes
: : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
#__inference__wrapped_model_28086766
input_1F
2generator_1_dense_1_matmul_readvariableop_resource:
??bB
3generator_1_dense_1_biasadd_readvariableop_resource:	?bO
3generator_1_conv2d_6_conv2d_readvariableop_resource:??C
4generator_1_conv2d_6_biasadd_readvariableop_resource:	?M
3generator_1_conv2d_7_conv2d_readvariableop_resource:@@B
4generator_1_conv2d_7_biasadd_readvariableop_resource:@M
3generator_1_conv2d_8_conv2d_readvariableop_resource: B
4generator_1_conv2d_8_biasadd_readvariableop_resource:
identity??+generator_1/conv2d_6/BiasAdd/ReadVariableOp?*generator_1/conv2d_6/Conv2D/ReadVariableOp?+generator_1/conv2d_7/BiasAdd/ReadVariableOp?*generator_1/conv2d_7/Conv2D/ReadVariableOp?+generator_1/conv2d_8/BiasAdd/ReadVariableOp?*generator_1/conv2d_8/Conv2D/ReadVariableOp?*generator_1/dense_1/BiasAdd/ReadVariableOp?)generator_1/dense_1/MatMul/ReadVariableOp?
)generator_1/dense_1/MatMul/ReadVariableOpReadVariableOp2generator_1_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02+
)generator_1/dense_1/MatMul/ReadVariableOp?
generator_1/dense_1/MatMulMatMulinput_11generator_1/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
generator_1/dense_1/MatMul?
*generator_1/dense_1/BiasAdd/ReadVariableOpReadVariableOp3generator_1_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02,
*generator_1/dense_1/BiasAdd/ReadVariableOp?
generator_1/dense_1/BiasAddBiasAdd$generator_1/dense_1/MatMul:product:02generator_1/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
generator_1/dense_1/BiasAdd?
%generator_1/glu_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%generator_1/glu_1/strided_slice/stack?
'generator_1/glu_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'generator_1/glu_1/strided_slice/stack_1?
'generator_1/glu_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'generator_1/glu_1/strided_slice/stack_2?
generator_1/glu_1/strided_sliceStridedSlice$generator_1/dense_1/BiasAdd:output:0.generator_1/glu_1/strided_slice/stack:output:00generator_1/glu_1/strided_slice/stack_1:output:00generator_1/glu_1/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2!
generator_1/glu_1/strided_slice?
'generator_1/glu_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'generator_1/glu_1/strided_slice_1/stack?
)generator_1/glu_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)generator_1/glu_1/strided_slice_1/stack_1?
)generator_1/glu_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)generator_1/glu_1/strided_slice_1/stack_2?
!generator_1/glu_1/strided_slice_1StridedSlice$generator_1/dense_1/BiasAdd:output:00generator_1/glu_1/strided_slice_1/stack:output:02generator_1/glu_1/strided_slice_1/stack_1:output:02generator_1/glu_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2#
!generator_1/glu_1/strided_slice_1?
generator_1/glu_1/SigmoidSigmoid*generator_1/glu_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
generator_1/glu_1/Sigmoid?
generator_1/glu_1/mulMul(generator_1/glu_1/strided_slice:output:0generator_1/glu_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
generator_1/glu_1/mul?
generator_1/reshape_1/ShapeShapegenerator_1/glu_1/mul:z:0*
T0*
_output_shapes
:2
generator_1/reshape_1/Shape?
)generator_1/reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)generator_1/reshape_1/strided_slice/stack?
+generator_1/reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+generator_1/reshape_1/strided_slice/stack_1?
+generator_1/reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+generator_1/reshape_1/strided_slice/stack_2?
#generator_1/reshape_1/strided_sliceStridedSlice$generator_1/reshape_1/Shape:output:02generator_1/reshape_1/strided_slice/stack:output:04generator_1/reshape_1/strided_slice/stack_1:output:04generator_1/reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#generator_1/reshape_1/strided_slice?
%generator_1/reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%generator_1/reshape_1/Reshape/shape/1?
%generator_1/reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%generator_1/reshape_1/Reshape/shape/2?
%generator_1/reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2'
%generator_1/reshape_1/Reshape/shape/3?
#generator_1/reshape_1/Reshape/shapePack,generator_1/reshape_1/strided_slice:output:0.generator_1/reshape_1/Reshape/shape/1:output:0.generator_1/reshape_1/Reshape/shape/2:output:0.generator_1/reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#generator_1/reshape_1/Reshape/shape?
generator_1/reshape_1/ReshapeReshapegenerator_1/glu_1/mul:z:0,generator_1/reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
generator_1/reshape_1/Reshape?
!generator_1/up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!generator_1/up_sampling2d_1/Const?
#generator_1/up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator_1/up_sampling2d_1/Const_1?
generator_1/up_sampling2d_1/mulMul*generator_1/up_sampling2d_1/Const:output:0,generator_1/up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2!
generator_1/up_sampling2d_1/mul?
8generator_1/up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighbor&generator_1/reshape_1/Reshape:output:0#generator_1/up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2:
8generator_1/up_sampling2d_1/resize/ResizeNearestNeighbor?
*generator_1/conv2d_6/Conv2D/ReadVariableOpReadVariableOp3generator_1_conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02,
*generator_1/conv2d_6/Conv2D/ReadVariableOp?
generator_1/conv2d_6/Conv2DConv2DIgenerator_1/up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:02generator_1/conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
generator_1/conv2d_6/Conv2D?
+generator_1/conv2d_6/BiasAdd/ReadVariableOpReadVariableOp4generator_1_conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02-
+generator_1/conv2d_6/BiasAdd/ReadVariableOp?
generator_1/conv2d_6/BiasAddBiasAdd$generator_1/conv2d_6/Conv2D:output:03generator_1/conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
generator_1/conv2d_6/BiasAdd?
'generator_1/glu_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2)
'generator_1/glu_1/strided_slice_2/stack?
)generator_1/glu_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2+
)generator_1/glu_1/strided_slice_2/stack_1?
)generator_1/glu_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_1/glu_1/strided_slice_2/stack_2?
!generator_1/glu_1/strided_slice_2StridedSlice%generator_1/conv2d_6/BiasAdd:output:00generator_1/glu_1/strided_slice_2/stack:output:02generator_1/glu_1/strided_slice_2/stack_1:output:02generator_1/glu_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2#
!generator_1/glu_1/strided_slice_2?
'generator_1/glu_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2)
'generator_1/glu_1/strided_slice_3/stack?
)generator_1/glu_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2+
)generator_1/glu_1/strided_slice_3/stack_1?
)generator_1/glu_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_1/glu_1/strided_slice_3/stack_2?
!generator_1/glu_1/strided_slice_3StridedSlice%generator_1/conv2d_6/BiasAdd:output:00generator_1/glu_1/strided_slice_3/stack:output:02generator_1/glu_1/strided_slice_3/stack_1:output:02generator_1/glu_1/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2#
!generator_1/glu_1/strided_slice_3?
generator_1/glu_1/Sigmoid_1Sigmoid*generator_1/glu_1/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
generator_1/glu_1/Sigmoid_1?
generator_1/glu_1/mul_1Mul*generator_1/glu_1/strided_slice_2:output:0generator_1/glu_1/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
generator_1/glu_1/mul_1?
#generator_1/up_sampling2d_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator_1/up_sampling2d_1/Const_2?
#generator_1/up_sampling2d_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator_1/up_sampling2d_1/Const_3?
!generator_1/up_sampling2d_1/mul_1Mul,generator_1/up_sampling2d_1/Const_2:output:0,generator_1/up_sampling2d_1/Const_3:output:0*
T0*
_output_shapes
:2#
!generator_1/up_sampling2d_1/mul_1?
:generator_1/up_sampling2d_1/resize_1/ResizeNearestNeighborResizeNearestNeighborgenerator_1/glu_1/mul_1:z:0%generator_1/up_sampling2d_1/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2<
:generator_1/up_sampling2d_1/resize_1/ResizeNearestNeighbor?
*generator_1/conv2d_7/Conv2D/ReadVariableOpReadVariableOp3generator_1_conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02,
*generator_1/conv2d_7/Conv2D/ReadVariableOp?
generator_1/conv2d_7/Conv2DConv2DKgenerator_1/up_sampling2d_1/resize_1/ResizeNearestNeighbor:resized_images:02generator_1/conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
generator_1/conv2d_7/Conv2D?
+generator_1/conv2d_7/BiasAdd/ReadVariableOpReadVariableOp4generator_1_conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02-
+generator_1/conv2d_7/BiasAdd/ReadVariableOp?
generator_1/conv2d_7/BiasAddBiasAdd$generator_1/conv2d_7/Conv2D:output:03generator_1/conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
generator_1/conv2d_7/BiasAdd?
'generator_1/glu_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2)
'generator_1/glu_1/strided_slice_4/stack?
)generator_1/glu_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2+
)generator_1/glu_1/strided_slice_4/stack_1?
)generator_1/glu_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_1/glu_1/strided_slice_4/stack_2?
!generator_1/glu_1/strided_slice_4StridedSlice%generator_1/conv2d_7/BiasAdd:output:00generator_1/glu_1/strided_slice_4/stack:output:02generator_1/glu_1/strided_slice_4/stack_1:output:02generator_1/glu_1/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2#
!generator_1/glu_1/strided_slice_4?
'generator_1/glu_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2)
'generator_1/glu_1/strided_slice_5/stack?
)generator_1/glu_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2+
)generator_1/glu_1/strided_slice_5/stack_1?
)generator_1/glu_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_1/glu_1/strided_slice_5/stack_2?
!generator_1/glu_1/strided_slice_5StridedSlice%generator_1/conv2d_7/BiasAdd:output:00generator_1/glu_1/strided_slice_5/stack:output:02generator_1/glu_1/strided_slice_5/stack_1:output:02generator_1/glu_1/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2#
!generator_1/glu_1/strided_slice_5?
generator_1/glu_1/Sigmoid_2Sigmoid*generator_1/glu_1/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
generator_1/glu_1/Sigmoid_2?
generator_1/glu_1/mul_2Mul*generator_1/glu_1/strided_slice_4:output:0generator_1/glu_1/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
generator_1/glu_1/mul_2?
*generator_1/conv2d_8/Conv2D/ReadVariableOpReadVariableOp3generator_1_conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02,
*generator_1/conv2d_8/Conv2D/ReadVariableOp?
generator_1/conv2d_8/Conv2DConv2Dgenerator_1/glu_1/mul_2:z:02generator_1/conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
generator_1/conv2d_8/Conv2D?
+generator_1/conv2d_8/BiasAdd/ReadVariableOpReadVariableOp4generator_1_conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02-
+generator_1/conv2d_8/BiasAdd/ReadVariableOp?
generator_1/conv2d_8/BiasAddBiasAdd$generator_1/conv2d_8/Conv2D:output:03generator_1/conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
generator_1/conv2d_8/BiasAdd?
generator_1/conv2d_8/TanhTanh%generator_1/conv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
generator_1/conv2d_8/Tanh?
IdentityIdentitygenerator_1/conv2d_8/Tanh:y:0,^generator_1/conv2d_6/BiasAdd/ReadVariableOp+^generator_1/conv2d_6/Conv2D/ReadVariableOp,^generator_1/conv2d_7/BiasAdd/ReadVariableOp+^generator_1/conv2d_7/Conv2D/ReadVariableOp,^generator_1/conv2d_8/BiasAdd/ReadVariableOp+^generator_1/conv2d_8/Conv2D/ReadVariableOp+^generator_1/dense_1/BiasAdd/ReadVariableOp*^generator_1/dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2Z
+generator_1/conv2d_6/BiasAdd/ReadVariableOp+generator_1/conv2d_6/BiasAdd/ReadVariableOp2X
*generator_1/conv2d_6/Conv2D/ReadVariableOp*generator_1/conv2d_6/Conv2D/ReadVariableOp2Z
+generator_1/conv2d_7/BiasAdd/ReadVariableOp+generator_1/conv2d_7/BiasAdd/ReadVariableOp2X
*generator_1/conv2d_7/Conv2D/ReadVariableOp*generator_1/conv2d_7/Conv2D/ReadVariableOp2Z
+generator_1/conv2d_8/BiasAdd/ReadVariableOp+generator_1/conv2d_8/BiasAdd/ReadVariableOp2X
*generator_1/conv2d_8/Conv2D/ReadVariableOp*generator_1/conv2d_8/Conv2D/ReadVariableOp2X
*generator_1/dense_1/BiasAdd/ReadVariableOp*generator_1/dense_1/BiasAdd/ReadVariableOp2V
)generator_1/dense_1/MatMul/ReadVariableOp)generator_1/dense_1/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
+__inference_conv2d_7_layer_call_fn_28087856

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_280869462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
F__inference_conv2d_6_layer_call_and_return_conditional_losses_28087828

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28087755

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+??????????????????????????? *

begin_mask*
end_mask2
strided_slice?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+??????????????????????????? *

begin_mask*
end_mask2
strided_slice_1?
SigmoidSigmoidstrided_slice_1:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
Sigmoid?
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_28087676

inputs2
matmul_readvariableop_resource:
??b.
biasadd_readvariableop_resource:	?b
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28087699

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice_1j
SigmoidSigmoidstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????12	
Sigmoidi
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?
E
(__inference_glu_1_layer_call_fn_28086787
input_1
identity?
PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280867842
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:Q M
(
_output_shapes
:??????????b
!
_user_specified_name	input_1
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28086965

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+??????????????????????????? *

begin_mask*
end_mask2
strided_slice?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+??????????????????????????? *

begin_mask*
end_mask2
strided_slice_1?
SigmoidSigmoidstrided_slice_1:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
Sigmoid?
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?	
?
.__inference_generator_1_layer_call_fn_28087645

inputs
unknown:
??b
	unknown_0:	?b%
	unknown_1:??
	unknown_2:	?#
	unknown_3:@@
	unknown_4:@#
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_generator_1_layer_call_and_return_conditional_losses_280871512
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_7_layer_call_and_return_conditional_losses_28086946

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+???????????????????????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28087727

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+???????????????????????????@*

begin_mask*
end_mask2
strided_slice?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+???????????????????????????@*

begin_mask*
end_mask2
strided_slice_1?
SigmoidSigmoidstrided_slice_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
Sigmoid?
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_glu_1_layer_call_fn_28087784

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280869652
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
?
F__inference_conv2d_6_layer_call_and_return_conditional_losses_28086914

inputs:
conv2d_readvariableop_resource:??.
biasadd_readvariableop_resource:	?
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*E
_input_shapes4
2:,????????????????????????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
F__inference_conv2d_8_layer_call_and_return_conditional_losses_28087867

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource:
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAddr
TanhTanhBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????2
Tanh?
IdentityIdentityTanh:y:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
c
G__inference_reshape_1_layer_call_and_return_conditional_losses_28087813

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_sliced
Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/1d
Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
Reshape/shape/2e
Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
Reshape/shape/3?
Reshape/shapePackstrided_slice:output:0Reshape/shape/1:output:0Reshape/shape/2:output:0Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
Reshape/shapex
ReshapeReshapeinputsReshape/shape:output:0*
T0*0
_output_shapes
:??????????2	
Reshapem
IdentityIdentityReshape:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????1:P L
(
_output_shapes
:??????????1
 
_user_specified_nameinputs
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28086808

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice_1j
SigmoidSigmoidstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????12	
Sigmoidi
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?
D
(__inference_glu_1_layer_call_fn_28087789

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280870352
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?
N
2__inference_up_sampling2d_1_layer_call_fn_28086863

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_280868572
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28086933

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+???????????????????????????@*

begin_mask*
end_mask2
strided_slice?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+???????????????????????????@*

begin_mask*
end_mask2
strided_slice_1?
SigmoidSigmoidstrided_slice_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
Sigmoid?
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28087769

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+???????????????????????????@*

begin_mask*
end_mask2
strided_slice?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+???????????????????????????@*

begin_mask*
end_mask2
strided_slice_1?
SigmoidSigmoidstrided_slice_1:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
Sigmoid?
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
i
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_28086857

inputs
identityD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
:2
strided_slice_
ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
Const^
mulMulstrided_slice:output:0Const:output:0*
T0*
_output_shapes
:2
mul?
resize/ResizeNearestNeighborResizeNearestNeighborinputsmul:z:0*
T0*J
_output_shapes8
6:4????????????????????????????????????*
half_pixel_centers(2
resize/ResizeNearestNeighbor?
IdentityIdentity-resize/ResizeNearestNeighbor:resized_images:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_glu_1_layer_call_fn_28087774

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280867842
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?
?
+__inference_conv2d_8_layer_call_fn_28087876

inputs!
unknown: 
	unknown_0:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_280869782
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*D
_input_shapes3
1:+??????????????????????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
!__inference__traced_save_28087923
file_prefix-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop.
*savev2_conv2d_6_kernel_read_readvariableop,
(savev2_conv2d_6_bias_read_readvariableop.
*savev2_conv2d_7_kernel_read_readvariableop,
(savev2_conv2d_7_bias_read_readvariableop.
*savev2_conv2d_8_kernel_read_readvariableop,
(savev2_conv2d_8_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*?
value?B?	B'dense/kernel/.ATTRIBUTES/VARIABLE_VALUEB%dense/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop*savev2_conv2d_6_kernel_read_readvariableop(savev2_conv2d_6_bias_read_readvariableop*savev2_conv2d_7_kernel_read_readvariableop(savev2_conv2d_7_bias_read_readvariableop*savev2_conv2d_8_kernel_read_readvariableop(savev2_conv2d_8_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*u
_input_shapesd
b: :
??b:?b:??:?:@@:@: :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
??b:!

_output_shapes	
:?b:.*
(
_output_shapes
:??:!

_output_shapes	
:?:,(
&
_output_shapes
:@@: 

_output_shapes
:@:,(
&
_output_shapes
: : 

_output_shapes
::	

_output_shapes
: 
?
E
(__inference_glu_1_layer_call_fn_28086816
input_1
identity?
PartitionedCallPartitionedCallinput_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280868082
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:Q M
(
_output_shapes
:??????????b
!
_user_specified_name	input_1
?	
?
E__inference_dense_1_layer_call_and_return_conditional_losses_28086880

inputs2
matmul_readvariableop_resource:
??b.
biasadd_readvariableop_resource:	?b
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
&__inference_signature_wrapper_28087274
input_1
unknown:
??b
	unknown_0:	?b%
	unknown_1:??
	unknown_2:	?#
	unknown_3:@@
	unknown_4:@#
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *,
f'R%
#__inference__wrapped_model_280867662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28087713

inputs
identity{
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice/stack
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice/stack_1
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
strided_slice_1j
SigmoidSigmoidstrided_slice_1:output:0*
T0*(
_output_shapes
:??????????12	
Sigmoidi
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
mul\
IdentityIdentitymul:z:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?h
?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087505
input_1:
&dense_1_matmul_readvariableop_resource:
??b6
'dense_1_biasadd_readvariableop_resource:	?bC
'conv2d_6_conv2d_readvariableop_resource:??7
(conv2d_6_biasadd_readvariableop_resource:	?A
'conv2d_7_conv2d_readvariableop_resource:@@6
(conv2d_7_biasadd_readvariableop_resource:@A
'conv2d_8_conv2d_readvariableop_resource: 6
(conv2d_8_biasadd_readvariableop_resource:
identity??conv2d_6/BiasAdd/ReadVariableOp?conv2d_6/Conv2D/ReadVariableOp?conv2d_7/BiasAdd/ReadVariableOp?conv2d_7/Conv2D/ReadVariableOp?conv2d_8/BiasAdd/ReadVariableOp?conv2d_8/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulinput_1%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_1/BiasAdd?
glu_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_1/strided_slice/stack?
glu_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_1/strided_slice/stack_1?
glu_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_1/strided_slice/stack_2?
glu_1/strided_sliceStridedSlicedense_1/BiasAdd:output:0"glu_1/strided_slice/stack:output:0$glu_1/strided_slice/stack_1:output:0$glu_1/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_1/strided_slice?
glu_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_1/strided_slice_1/stack?
glu_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_1/strided_slice_1/stack_1?
glu_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_1/strided_slice_1/stack_2?
glu_1/strided_slice_1StridedSlicedense_1/BiasAdd:output:0$glu_1/strided_slice_1/stack:output:0&glu_1/strided_slice_1/stack_1:output:0&glu_1/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_1/strided_slice_1|
glu_1/SigmoidSigmoidglu_1/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_1/Sigmoid?
	glu_1/mulMulglu_1/strided_slice:output:0glu_1/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_1/mul_
reshape_1/ShapeShapeglu_1/mul:z:0*
T0*
_output_shapes
:2
reshape_1/Shape?
reshape_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_1/strided_slice/stack?
reshape_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_1?
reshape_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_1/strided_slice/stack_2?
reshape_1/strided_sliceStridedSlicereshape_1/Shape:output:0&reshape_1/strided_slice/stack:output:0(reshape_1/strided_slice/stack_1:output:0(reshape_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_1/strided_slicex
reshape_1/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/1x
reshape_1/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_1/Reshape/shape/2y
reshape_1/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_1/Reshape/shape/3?
reshape_1/Reshape/shapePack reshape_1/strided_slice:output:0"reshape_1/Reshape/shape/1:output:0"reshape_1/Reshape/shape/2:output:0"reshape_1/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_1/Reshape/shape?
reshape_1/ReshapeReshapeglu_1/mul:z:0 reshape_1/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_1/Reshape
up_sampling2d_1/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const?
up_sampling2d_1/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_1?
up_sampling2d_1/mulMulup_sampling2d_1/Const:output:0 up_sampling2d_1/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul?
,up_sampling2d_1/resize/ResizeNearestNeighborResizeNearestNeighborreshape_1/Reshape:output:0up_sampling2d_1/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_1/resize/ResizeNearestNeighbor?
conv2d_6/Conv2D/ReadVariableOpReadVariableOp'conv2d_6_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_6/Conv2D/ReadVariableOp?
conv2d_6/Conv2DConv2D=up_sampling2d_1/resize/ResizeNearestNeighbor:resized_images:0&conv2d_6/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_6/Conv2D?
conv2d_6/BiasAdd/ReadVariableOpReadVariableOp(conv2d_6_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_6/BiasAdd/ReadVariableOp?
conv2d_6/BiasAddBiasAddconv2d_6/Conv2D:output:0'conv2d_6/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_6/BiasAdd?
glu_1/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_2/stack?
glu_1/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_1/strided_slice_2/stack_1?
glu_1/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_2/stack_2?
glu_1/strided_slice_2StridedSliceconv2d_6/BiasAdd:output:0$glu_1/strided_slice_2/stack:output:0&glu_1/strided_slice_2/stack_1:output:0&glu_1/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_1/strided_slice_2?
glu_1/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_1/strided_slice_3/stack?
glu_1/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_3/stack_1?
glu_1/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_3/stack_2?
glu_1/strided_slice_3StridedSliceconv2d_6/BiasAdd:output:0$glu_1/strided_slice_3/stack:output:0&glu_1/strided_slice_3/stack_1:output:0&glu_1/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_1/strided_slice_3?
glu_1/Sigmoid_1Sigmoidglu_1/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_1/Sigmoid_1?
glu_1/mul_1Mulglu_1/strided_slice_2:output:0glu_1/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_1/mul_1?
up_sampling2d_1/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_2?
up_sampling2d_1/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_1/Const_3?
up_sampling2d_1/mul_1Mul up_sampling2d_1/Const_2:output:0 up_sampling2d_1/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_1/mul_1?
.up_sampling2d_1/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_1/mul_1:z:0up_sampling2d_1/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_1/resize_1/ResizeNearestNeighbor?
conv2d_7/Conv2D/ReadVariableOpReadVariableOp'conv2d_7_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_7/Conv2D/ReadVariableOp?
conv2d_7/Conv2DConv2D?up_sampling2d_1/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_7/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_7/Conv2D?
conv2d_7/BiasAdd/ReadVariableOpReadVariableOp(conv2d_7_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_7/BiasAdd/ReadVariableOp?
conv2d_7/BiasAddBiasAddconv2d_7/Conv2D:output:0'conv2d_7/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_7/BiasAdd?
glu_1/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_4/stack?
glu_1/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_4/stack_1?
glu_1/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_4/stack_2?
glu_1/strided_slice_4StridedSliceconv2d_7/BiasAdd:output:0$glu_1/strided_slice_4/stack:output:0&glu_1/strided_slice_4/stack_1:output:0&glu_1/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_1/strided_slice_4?
glu_1/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_5/stack?
glu_1/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_1/strided_slice_5/stack_1?
glu_1/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_1/strided_slice_5/stack_2?
glu_1/strided_slice_5StridedSliceconv2d_7/BiasAdd:output:0$glu_1/strided_slice_5/stack:output:0&glu_1/strided_slice_5/stack_1:output:0&glu_1/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_1/strided_slice_5?
glu_1/Sigmoid_2Sigmoidglu_1/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_1/Sigmoid_2?
glu_1/mul_2Mulglu_1/strided_slice_4:output:0glu_1/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_1/mul_2?
conv2d_8/Conv2D/ReadVariableOpReadVariableOp'conv2d_8_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_8/Conv2D/ReadVariableOp?
conv2d_8/Conv2DConv2Dglu_1/mul_2:z:0&conv2d_8/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_8/Conv2D?
conv2d_8/BiasAdd/ReadVariableOpReadVariableOp(conv2d_8_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_8/BiasAdd/ReadVariableOp?
conv2d_8/BiasAddBiasAddconv2d_8/Conv2D:output:0'conv2d_8/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_8/BiasAdd{
conv2d_8/TanhTanhconv2d_8/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_8/Tanh?
IdentityIdentityconv2d_8/Tanh:y:0 ^conv2d_6/BiasAdd/ReadVariableOp^conv2d_6/Conv2D/ReadVariableOp ^conv2d_7/BiasAdd/ReadVariableOp^conv2d_7/Conv2D/ReadVariableOp ^conv2d_8/BiasAdd/ReadVariableOp^conv2d_8/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2B
conv2d_6/BiasAdd/ReadVariableOpconv2d_6/BiasAdd/ReadVariableOp2@
conv2d_6/Conv2D/ReadVariableOpconv2d_6/Conv2D/ReadVariableOp2B
conv2d_7/BiasAdd/ReadVariableOpconv2d_7/BiasAdd/ReadVariableOp2@
conv2d_7/Conv2D/ReadVariableOpconv2d_7/Conv2D/ReadVariableOp2B
conv2d_8/BiasAdd/ReadVariableOpconv2d_8/BiasAdd/ReadVariableOp2@
conv2d_8/Conv2D/ReadVariableOpconv2d_8/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?+
?
I__inference_generator_1_layer_call_and_return_conditional_losses_28086985

inputs$
dense_1_28086881:
??b
dense_1_28086883:	?b-
conv2d_6_28086915:?? 
conv2d_6_28086917:	?+
conv2d_7_28086947:@@
conv2d_7_28086949:@+
conv2d_8_28086979: 
conv2d_8_28086981:
identity?? conv2d_6/StatefulPartitionedCall? conv2d_7/StatefulPartitionedCall? conv2d_8/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1_28086881dense_1_28086883*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *N
fIRG
E__inference_dense_1_layer_call_and_return_conditional_losses_280868802!
dense_1/StatefulPartitionedCall?
glu_1/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280867842
glu_1/PartitionedCall?
reshape_1/PartitionedCallPartitionedCallglu_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_reshape_1_layer_call_and_return_conditional_losses_280869012
reshape_1/PartitionedCall?
up_sampling2d_1/PartitionedCallPartitionedCall"reshape_1/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_280868572!
up_sampling2d_1/PartitionedCall?
 conv2d_6/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_1/PartitionedCall:output:0conv2d_6_28086915conv2d_6_28086917*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_6_layer_call_and_return_conditional_losses_280869142"
 conv2d_6/StatefulPartitionedCall?
glu_1/PartitionedCall_1PartitionedCall)conv2d_6/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280869332
glu_1/PartitionedCall_1?
!up_sampling2d_1/PartitionedCall_1PartitionedCall glu_1/PartitionedCall_1:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *V
fQRO
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_280868572#
!up_sampling2d_1/PartitionedCall_1?
 conv2d_7/StatefulPartitionedCallStatefulPartitionedCall*up_sampling2d_1/PartitionedCall_1:output:0conv2d_7_28086947conv2d_7_28086949*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_7_layer_call_and_return_conditional_losses_280869462"
 conv2d_7/StatefulPartitionedCall?
glu_1/PartitionedCall_2PartitionedCall)conv2d_7/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? * 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280869652
glu_1/PartitionedCall_2?
 conv2d_8/StatefulPartitionedCallStatefulPartitionedCall glu_1/PartitionedCall_2:output:0conv2d_8_28086979conv2d_8_28086981*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_8_layer_call_and_return_conditional_losses_280869782"
 conv2d_8/StatefulPartitionedCall?
IdentityIdentity)conv2d_8/StatefulPartitionedCall:output:0!^conv2d_6/StatefulPartitionedCall!^conv2d_7/StatefulPartitionedCall!^conv2d_8/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_6/StatefulPartitionedCall conv2d_6/StatefulPartitionedCall2D
 conv2d_7/StatefulPartitionedCall conv2d_7/StatefulPartitionedCall2D
 conv2d_8/StatefulPartitionedCall conv2d_8/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
D
(__inference_glu_1_layer_call_fn_28087794

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280869332
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*A
_input_shapes0
.:,????????????????????????????:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
D
(__inference_glu_1_layer_call_fn_28087779

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????1* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *L
fGRE
C__inference_glu_1_layer_call_and_return_conditional_losses_280868082
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*'
_input_shapes
:??????????b:P L
(
_output_shapes
:??????????b
 
_user_specified_nameinputs
?	
?
.__inference_generator_1_layer_call_fn_28087603
input_1
unknown:
??b
	unknown_0:	?b%
	unknown_1:??
	unknown_2:	?#
	unknown_3:@@
	unknown_4:@#
	unknown_5: 
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????**
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *R
fMRK
I__inference_generator_1_layer_call_and_return_conditional_losses_280869852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
_
C__inference_glu_1_layer_call_and_return_conditional_losses_28087035

inputs
identity?
strided_slice/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack?
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice/stack_1?
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice/stack_2?
strided_sliceStridedSliceinputsstrided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+??????????????????????????? *

begin_mask*
end_mask2
strided_slice?
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack?
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
strided_slice_1/stack_1?
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
strided_slice_1/stack_2?
strided_slice_1StridedSliceinputsstrided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*A
_output_shapes/
-:+??????????????????????????? *

begin_mask*
end_mask2
strided_slice_1?
SigmoidSigmoidstrided_slice_1:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
Sigmoid?
mulMulstrided_slice:output:0Sigmoid:y:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
mulu
IdentityIdentitymul:z:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*@
_input_shapes/
-:+???????????????????????????@:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
<
input_11
serving_default_input_1:0??????????D
output_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
	dense

activation
reshape
upsample
	conv1
	conv2
	conv3
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
*Y&call_and_return_all_conditional_losses
Z__call__
[_default_save_signature"?
_tf_keras_model?{"name": "generator_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Generator", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [32, 128]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Generator"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__"?
_tf_keras_model?{"name": "glu_1", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "GLU", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 12544]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "GLU"}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"name": "reshape_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_1", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 128]}}, "shared_object_id": 4}
?
trainable_variables
regularization_losses
	variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_layer?{"name": "up_sampling2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_1", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 6}}
?


kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?	
_tf_keras_layer?	{"name": "conv2d_6", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_6", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 128]}}
?


%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?	
_tf_keras_layer?	{"name": "conv2d_7", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_7", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 64]}}
?


+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
*h&call_and_return_all_conditional_losses
i__call__"?	
_tf_keras_layer?	{"name": "conv2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_8", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
X
0
1
2
 3
%4
&5
+6
,7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
0
1
2
 3
%4
&5
+6
,7"
trackable_list_wrapper
?

1layers
trainable_variables
2non_trainable_variables
	regularization_losses
3metrics
4layer_regularization_losses
5layer_metrics

	variables
Z__call__
[_default_save_signature
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
,
jserving_default"
signature_map
": 
??b2dense_1/kernel
:?b2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

6layers
trainable_variables
7non_trainable_variables
regularization_losses
8metrics
9layer_regularization_losses
:layer_metrics
	variables
]__call__
*\&call_and_return_all_conditional_losses
&\"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

;layers
trainable_variables
<non_trainable_variables
regularization_losses
=metrics
>layer_regularization_losses
?layer_metrics
	variables
___call__
*^&call_and_return_all_conditional_losses
&^"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

@layers
trainable_variables
Anon_trainable_variables
regularization_losses
Bmetrics
Clayer_regularization_losses
Dlayer_metrics
	variables
a__call__
*`&call_and_return_all_conditional_losses
&`"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

Elayers
trainable_variables
Fnon_trainable_variables
regularization_losses
Gmetrics
Hlayer_regularization_losses
Ilayer_metrics
	variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
+:)??2conv2d_6/kernel
:?2conv2d_6/bias
.
0
 1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?

Jlayers
!trainable_variables
Knon_trainable_variables
"regularization_losses
Lmetrics
Mlayer_regularization_losses
Nlayer_metrics
#	variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_7/kernel
:@2conv2d_7/bias
.
%0
&1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?

Olayers
'trainable_variables
Pnon_trainable_variables
(regularization_losses
Qmetrics
Rlayer_regularization_losses
Slayer_metrics
)	variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_8/kernel
:2conv2d_8/bias
.
+0
,1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?

Tlayers
-trainable_variables
Unon_trainable_variables
.regularization_losses
Vmetrics
Wlayer_regularization_losses
Xlayer_metrics
/	variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
Q
0
1
2
3
4
5
6"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?2?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087351
I__inference_generator_1_layer_call_and_return_conditional_losses_28087428
I__inference_generator_1_layer_call_and_return_conditional_losses_28087505
I__inference_generator_1_layer_call_and_return_conditional_losses_28087582?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_generator_1_layer_call_fn_28087603
.__inference_generator_1_layer_call_fn_28087624
.__inference_generator_1_layer_call_fn_28087645
.__inference_generator_1_layer_call_fn_28087666?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_28086766?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *'?$
"?
input_1??????????
?2?
E__inference_dense_1_layer_call_and_return_conditional_losses_28087676?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_dense_1_layer_call_fn_28087685?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
C__inference_glu_1_layer_call_and_return_conditional_losses_28087699
C__inference_glu_1_layer_call_and_return_conditional_losses_28087713
C__inference_glu_1_layer_call_and_return_conditional_losses_28086830
C__inference_glu_1_layer_call_and_return_conditional_losses_28086844
C__inference_glu_1_layer_call_and_return_conditional_losses_28087727
C__inference_glu_1_layer_call_and_return_conditional_losses_28087741
C__inference_glu_1_layer_call_and_return_conditional_losses_28087755
C__inference_glu_1_layer_call_and_return_conditional_losses_28087769?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
(__inference_glu_1_layer_call_fn_28086787
(__inference_glu_1_layer_call_fn_28087774
(__inference_glu_1_layer_call_fn_28087779
(__inference_glu_1_layer_call_fn_28086816
(__inference_glu_1_layer_call_fn_28087784
(__inference_glu_1_layer_call_fn_28087789
(__inference_glu_1_layer_call_fn_28087794
(__inference_glu_1_layer_call_fn_28087799?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
G__inference_reshape_1_layer_call_and_return_conditional_losses_28087813?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_reshape_1_layer_call_fn_28087818?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_28086857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
2__inference_up_sampling2d_1_layer_call_fn_28086863?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
F__inference_conv2d_6_layer_call_and_return_conditional_losses_28087828?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_6_layer_call_fn_28087837?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_7_layer_call_and_return_conditional_losses_28087847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_7_layer_call_fn_28087856?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_8_layer_call_and_return_conditional_losses_28087867?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
+__inference_conv2d_8_layer_call_fn_28087876?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
&__inference_signature_wrapper_28087274input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_28086766z %&+,1?.
'?$
"?
input_1??????????
? ";?8
6
output_1*?'
output_1??????????
F__inference_conv2d_6_layer_call_and_return_conditional_losses_28087828? J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
+__inference_conv2d_6_layer_call_fn_28087837? J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
F__inference_conv2d_7_layer_call_and_return_conditional_losses_28087847?%&I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_conv2d_7_layer_call_fn_28087856?%&I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
F__inference_conv2d_8_layer_call_and_return_conditional_losses_28087867?+,I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
+__inference_conv2d_8_layer_call_fn_28087876?+,I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
E__inference_dense_1_layer_call_and_return_conditional_losses_28087676^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????b
? 
*__inference_dense_1_layer_call_fn_28087685Q0?-
&?#
!?
inputs??????????
? "???????????b?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087351o %&+,4?1
*?'
!?
inputs??????????
p 
? "-?*
#? 
0?????????
? ?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087428o %&+,4?1
*?'
!?
inputs??????????
p
? "-?*
#? 
0?????????
? ?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087505p %&+,5?2
+?(
"?
input_1??????????
p 
? "-?*
#? 
0?????????
? ?
I__inference_generator_1_layer_call_and_return_conditional_losses_28087582p %&+,5?2
+?(
"?
input_1??????????
p
? "-?*
#? 
0?????????
? ?
.__inference_generator_1_layer_call_fn_28087603u %&+,5?2
+?(
"?
input_1??????????
p 
? "2?/+????????????????????????????
.__inference_generator_1_layer_call_fn_28087624t %&+,4?1
*?'
!?
inputs??????????
p 
? "2?/+????????????????????????????
.__inference_generator_1_layer_call_fn_28087645t %&+,4?1
*?'
!?
inputs??????????
p
? "2?/+????????????????????????????
.__inference_generator_1_layer_call_fn_28087666u %&+,5?2
+?(
"?
input_1??????????
p
? "2?/+????????????????????????????
C__inference_glu_1_layer_call_and_return_conditional_losses_28086830kA?>
'?$
"?
input_1??????????b
?

trainingp "&?#
?
0??????????1
? ?
C__inference_glu_1_layer_call_and_return_conditional_losses_28086844kA?>
'?$
"?
input_1??????????b
?

trainingp"&?#
?
0??????????1
? ?
C__inference_glu_1_layer_call_and_return_conditional_losses_28087699j@?=
&?#
!?
inputs??????????b
?

trainingp "&?#
?
0??????????1
? ?
C__inference_glu_1_layer_call_and_return_conditional_losses_28087713j@?=
&?#
!?
inputs??????????b
?

trainingp"&?#
?
0??????????1
? ?
C__inference_glu_1_layer_call_and_return_conditional_losses_28087727?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp "??<
5?2
0+???????????????????????????@
? ?
C__inference_glu_1_layer_call_and_return_conditional_losses_28087741?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp "??<
5?2
0+??????????????????????????? 
? ?
C__inference_glu_1_layer_call_and_return_conditional_losses_28087755?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp"??<
5?2
0+??????????????????????????? 
? ?
C__inference_glu_1_layer_call_and_return_conditional_losses_28087769?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp"??<
5?2
0+???????????????????????????@
? ?
(__inference_glu_1_layer_call_fn_28086787^A?>
'?$
"?
input_1??????????b
?

trainingp "???????????1?
(__inference_glu_1_layer_call_fn_28086816^A?>
'?$
"?
input_1??????????b
?

trainingp"???????????1?
(__inference_glu_1_layer_call_fn_28087774]@?=
&?#
!?
inputs??????????b
?

trainingp "???????????1?
(__inference_glu_1_layer_call_fn_28087779]@?=
&?#
!?
inputs??????????b
?

trainingp"???????????1?
(__inference_glu_1_layer_call_fn_28087784?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp "2?/+??????????????????????????? ?
(__inference_glu_1_layer_call_fn_28087789?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp"2?/+??????????????????????????? ?
(__inference_glu_1_layer_call_fn_28087794?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp "2?/+???????????????????????????@?
(__inference_glu_1_layer_call_fn_28087799?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp"2?/+???????????????????????????@?
G__inference_reshape_1_layer_call_and_return_conditional_losses_28087813b0?-
&?#
!?
inputs??????????1
? ".?+
$?!
0??????????
? ?
,__inference_reshape_1_layer_call_fn_28087818U0?-
&?#
!?
inputs??????????1
? "!????????????
&__inference_signature_wrapper_28087274? %&+,<?9
? 
2?/
-
input_1"?
input_1??????????";?8
6
output_1*?'
output_1??????????
M__inference_up_sampling2d_1_layer_call_and_return_conditional_losses_28086857?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
2__inference_up_sampling2d_1_layer_call_fn_28086863?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????