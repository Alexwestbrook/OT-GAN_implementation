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
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??b*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
??b*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?b*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:?b*
dtype0
?
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*
shared_nameconv2d/kernel
y
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*(
_output_shapes
:??*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:?*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_1/kernel
{
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_1/bias
k
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes
:@*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_2/kernel
{
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*&
_output_shapes
: *
dtype0
r
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_2/bias
k
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
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
regularization_losses
		variables

trainable_variables
	keras_api

signatures
h

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
R
regularization_losses
	variables
trainable_variables
	keras_api
h

kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
h

%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
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
1layer_regularization_losses
regularization_losses

2layers
3metrics
4layer_metrics
		variables

trainable_variables
5non_trainable_variables
 
IG
VARIABLE_VALUEdense/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUE
dense/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
6layer_regularization_losses
regularization_losses

7layers
8metrics
9layer_metrics
	variables
trainable_variables
:non_trainable_variables
 
 
 
?
;layer_regularization_losses
regularization_losses

<layers
=metrics
>layer_metrics
	variables
trainable_variables
?non_trainable_variables
 
 
 
?
@layer_regularization_losses
regularization_losses

Alayers
Bmetrics
Clayer_metrics
	variables
trainable_variables
Dnon_trainable_variables
 
 
 
?
Elayer_regularization_losses
regularization_losses

Flayers
Gmetrics
Hlayer_metrics
	variables
trainable_variables
Inon_trainable_variables
JH
VARIABLE_VALUEconv2d/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
FD
VARIABLE_VALUEconv2d/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
 1

0
 1
?
Jlayer_regularization_losses
!regularization_losses

Klayers
Lmetrics
Mlayer_metrics
"	variables
#trainable_variables
Nnon_trainable_variables
LJ
VARIABLE_VALUEconv2d_1/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_1/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

%0
&1

%0
&1
?
Olayer_regularization_losses
'regularization_losses

Players
Qmetrics
Rlayer_metrics
(	variables
)trainable_variables
Snon_trainable_variables
LJ
VARIABLE_VALUEconv2d_2/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_2/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
?
Tlayer_regularization_losses
-regularization_losses

Ulayers
Vmetrics
Wlayer_metrics
.	variables
/trainable_variables
Xnon_trainable_variables
 
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
|
serving_default_input_1Placeholder*(
_output_shapes
:??????????*
dtype0*
shape:??????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/bias*
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
&__inference_signature_wrapper_28847792
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_28848441
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasconv2d/kernelconv2d/biasconv2d_1/kernelconv2d_1/biasconv2d_2/kernelconv2d_2/bias*
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
$__inference__traced_restore_28848475??	
?
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_28848365

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
?e
?
G__inference_generator_layer_call_and_return_conditional_losses_28848100
input_18
$dense_matmul_readvariableop_resource:
??b4
%dense_biasadd_readvariableop_resource:	?bA
%conv2d_conv2d_readvariableop_resource:??5
&conv2d_biasadd_readvariableop_resource:	?A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinput_1#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense/BiasAdd?
glu/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu/strided_slice/stack?
glu/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu/strided_slice/stack_1?
glu/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu/strided_slice/stack_2?
glu/strided_sliceStridedSlicedense/BiasAdd:output:0 glu/strided_slice/stack:output:0"glu/strided_slice/stack_1:output:0"glu/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu/strided_slice?
glu/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu/strided_slice_1/stack?
glu/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu/strided_slice_1/stack_1?
glu/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu/strided_slice_1/stack_2?
glu/strided_slice_1StridedSlicedense/BiasAdd:output:0"glu/strided_slice_1/stack:output:0$glu/strided_slice_1/stack_1:output:0$glu/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu/strided_slice_1v
glu/SigmoidSigmoidglu/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu/Sigmoidy
glu/mulMulglu/strided_slice:output:0glu/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12	
glu/mulY
reshape/ShapeShapeglu/mul:z:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeglu/mul:z:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshape{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborreshape/Reshape:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAdd?
glu/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_2/stack?
glu/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu/strided_slice_2/stack_1?
glu/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_2/stack_2?
glu/strided_slice_2StridedSliceconv2d/BiasAdd:output:0"glu/strided_slice_2/stack:output:0$glu/strided_slice_2/stack_1:output:0$glu/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu/strided_slice_2?
glu/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu/strided_slice_3/stack?
glu/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_3/stack_1?
glu/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_3/stack_2?
glu/strided_slice_3StridedSliceconv2d/BiasAdd:output:0"glu/strided_slice_3/stack:output:0$glu/strided_slice_3/stack_1:output:0$glu/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu/strided_slice_3?
glu/Sigmoid_1Sigmoidglu/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu/Sigmoid_1?
	glu/mul_1Mulglu/strided_slice_2:output:0glu/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
	glu/mul_1
up_sampling2d/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_2
up_sampling2d/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_3?
up_sampling2d/mul_1Mulup_sampling2d/Const_2:output:0up_sampling2d/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul_1?
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborglu/mul_1:z:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,up_sampling2d/resize_1/ResizeNearestNeighbor?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
glu/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_4/stack?
glu/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_4/stack_1?
glu/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_4/stack_2?
glu/strided_slice_4StridedSliceconv2d_1/BiasAdd:output:0"glu/strided_slice_4/stack:output:0$glu/strided_slice_4/stack_1:output:0$glu/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu/strided_slice_4?
glu/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_5/stack?
glu/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_5/stack_1?
glu/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_5/stack_2?
glu/strided_slice_5StridedSliceconv2d_1/BiasAdd:output:0"glu/strided_slice_5/stack:output:0$glu/strided_slice_5/stack_1:output:0$glu/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu/strided_slice_5?
glu/Sigmoid_2Sigmoidglu/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu/Sigmoid_2?
	glu/mul_2Mulglu/strided_slice_4:output:0glu/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
	glu/mul_2?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dglu/mul_2:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/TanhTanhconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Tanh?
IdentityIdentityconv2d_2/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_28847496

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
?
L
0__inference_up_sampling2d_layer_call_fn_28847381

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
GPU2*0,1J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_288473752
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
]
A__inference_glu_layer_call_and_return_conditional_losses_28848273

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
?
]
A__inference_glu_layer_call_and_return_conditional_losses_28847553

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
,__inference_generator_layer_call_fn_28848121
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
GPU2*0,1J 8? *P
fKRI
G__inference_generator_layer_call_and_return_conditional_losses_288475032
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
?	
?
,__inference_generator_layer_call_fn_28848163

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
GPU2*0,1J 8? *P
fKRI
G__inference_generator_layer_call_and_return_conditional_losses_288476692
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
C__inference_dense_layer_call_and_return_conditional_losses_28848194

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
]
A__inference_glu_layer_call_and_return_conditional_losses_28848231

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
?	
?
C__inference_dense_layer_call_and_return_conditional_losses_28847398

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
?+
?
G__inference_generator_layer_call_and_return_conditional_losses_28847669

inputs"
dense_28847642:
??b
dense_28847644:	?b+
conv2d_28847650:??
conv2d_28847652:	?+
conv2d_1_28847657:@@
conv2d_1_28847659:@+
conv2d_2_28847663: 
conv2d_2_28847665:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_28847642dense_28847644*
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
GPU2*0,1J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_288473982
dense/StatefulPartitionedCall?
glu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288473262
glu/PartitionedCall?
reshape/PartitionedCallPartitionedCallglu/PartitionedCall:output:0*
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
GPU2*0,1J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_288474192
reshape/PartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
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
GPU2*0,1J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_288473752
up_sampling2d/PartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_28847650conv2d_28847652*
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
GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_288474322 
conv2d/StatefulPartitionedCall?
glu/PartitionedCall_1PartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288475872
glu/PartitionedCall_1?
up_sampling2d/PartitionedCall_1PartitionedCallglu/PartitionedCall_1:output:0*
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
GPU2*0,1J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_288473752!
up_sampling2d/PartitionedCall_1?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d/PartitionedCall_1:output:0conv2d_1_28847657conv2d_1_28847659*
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_288474642"
 conv2d_1/StatefulPartitionedCall?
glu/PartitionedCall_2PartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288475532
glu/PartitionedCall_2?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallglu/PartitionedCall_2:output:0conv2d_2_28847663conv2d_2_28847665*
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_288474962"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?e
?
G__inference_generator_layer_call_and_return_conditional_losses_28847869

inputs8
$dense_matmul_readvariableop_resource:
??b4
%dense_biasadd_readvariableop_resource:	?bA
%conv2d_conv2d_readvariableop_resource:??5
&conv2d_biasadd_readvariableop_resource:	?A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense/BiasAdd?
glu/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu/strided_slice/stack?
glu/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu/strided_slice/stack_1?
glu/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu/strided_slice/stack_2?
glu/strided_sliceStridedSlicedense/BiasAdd:output:0 glu/strided_slice/stack:output:0"glu/strided_slice/stack_1:output:0"glu/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu/strided_slice?
glu/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu/strided_slice_1/stack?
glu/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu/strided_slice_1/stack_1?
glu/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu/strided_slice_1/stack_2?
glu/strided_slice_1StridedSlicedense/BiasAdd:output:0"glu/strided_slice_1/stack:output:0$glu/strided_slice_1/stack_1:output:0$glu/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu/strided_slice_1v
glu/SigmoidSigmoidglu/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu/Sigmoidy
glu/mulMulglu/strided_slice:output:0glu/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12	
glu/mulY
reshape/ShapeShapeglu/mul:z:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeglu/mul:z:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshape{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborreshape/Reshape:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAdd?
glu/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_2/stack?
glu/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu/strided_slice_2/stack_1?
glu/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_2/stack_2?
glu/strided_slice_2StridedSliceconv2d/BiasAdd:output:0"glu/strided_slice_2/stack:output:0$glu/strided_slice_2/stack_1:output:0$glu/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu/strided_slice_2?
glu/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu/strided_slice_3/stack?
glu/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_3/stack_1?
glu/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_3/stack_2?
glu/strided_slice_3StridedSliceconv2d/BiasAdd:output:0"glu/strided_slice_3/stack:output:0$glu/strided_slice_3/stack_1:output:0$glu/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu/strided_slice_3?
glu/Sigmoid_1Sigmoidglu/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu/Sigmoid_1?
	glu/mul_1Mulglu/strided_slice_2:output:0glu/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
	glu/mul_1
up_sampling2d/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_2
up_sampling2d/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_3?
up_sampling2d/mul_1Mulup_sampling2d/Const_2:output:0up_sampling2d/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul_1?
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborglu/mul_1:z:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,up_sampling2d/resize_1/ResizeNearestNeighbor?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
glu/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_4/stack?
glu/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_4/stack_1?
glu/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_4/stack_2?
glu/strided_slice_4StridedSliceconv2d_1/BiasAdd:output:0"glu/strided_slice_4/stack:output:0$glu/strided_slice_4/stack_1:output:0$glu/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu/strided_slice_4?
glu/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_5/stack?
glu/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_5/stack_1?
glu/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_5/stack_2?
glu/strided_slice_5StridedSliceconv2d_1/BiasAdd:output:0"glu/strided_slice_5/stack:output:0$glu/strided_slice_5/stack_1:output:0$glu/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu/strided_slice_5?
glu/Sigmoid_2Sigmoidglu/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu/Sigmoid_2?
	glu/mul_2Mulglu/strided_slice_4:output:0glu/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
	glu/mul_2?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dglu/mul_2:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/TanhTanhconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Tanh?
IdentityIdentityconv2d_2/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
^
A__inference_glu_layer_call_and_return_conditional_losses_28847348
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
?+
?
G__inference_generator_layer_call_and_return_conditional_losses_28847503

inputs"
dense_28847399:
??b
dense_28847401:	?b+
conv2d_28847433:??
conv2d_28847435:	?+
conv2d_1_28847465:@@
conv2d_1_28847467:@+
conv2d_2_28847497: 
conv2d_2_28847499:
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?dense/StatefulPartitionedCall?
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_28847399dense_28847401*
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
GPU2*0,1J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_288473982
dense/StatefulPartitionedCall?
glu/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288473022
glu/PartitionedCall?
reshape/PartitionedCallPartitionedCallglu/PartitionedCall:output:0*
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
GPU2*0,1J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_288474192
reshape/PartitionedCall?
up_sampling2d/PartitionedCallPartitionedCall reshape/PartitionedCall:output:0*
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
GPU2*0,1J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_288473752
up_sampling2d/PartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCall&up_sampling2d/PartitionedCall:output:0conv2d_28847433conv2d_28847435*
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
GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_288474322 
conv2d/StatefulPartitionedCall?
glu/PartitionedCall_1PartitionedCall'conv2d/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288474512
glu/PartitionedCall_1?
up_sampling2d/PartitionedCall_1PartitionedCallglu/PartitionedCall_1:output:0*
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
GPU2*0,1J 8? *T
fORM
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_288473752!
up_sampling2d/PartitionedCall_1?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d/PartitionedCall_1:output:0conv2d_1_28847465conv2d_1_28847467*
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_288474642"
 conv2d_1/StatefulPartitionedCall?
glu/PartitionedCall_2PartitionedCall)conv2d_1/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288474832
glu/PartitionedCall_2?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCallglu/PartitionedCall_2:output:0conv2d_2_28847497conv2d_2_28847499*
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_288474962"
 conv2d_2/StatefulPartitionedCall?
IdentityIdentity)conv2d_2/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall^dense/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
]
A__inference_glu_layer_call_and_return_conditional_losses_28847326

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
?	
?
&__inference_signature_wrapper_28847792
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
#__inference__wrapped_model_288472842
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
?
]
A__inference_glu_layer_call_and_return_conditional_losses_28848259

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
?
F
*__inference_reshape_layer_call_fn_28848336

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
GPU2*0,1J 8? *N
fIRG
E__inference_reshape_layer_call_and_return_conditional_losses_288474192
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
?
]
A__inference_glu_layer_call_and_return_conditional_losses_28848287

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
?
!__inference__traced_save_28848441
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
]
A__inference_glu_layer_call_and_return_conditional_losses_28848217

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
?
?
D__inference_conv2d_layer_call_and_return_conditional_losses_28847432

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
?
C
&__inference_glu_layer_call_fn_28847334
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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288473262
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
?
B
&__inference_glu_layer_call_fn_28848307

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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288475532
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
a
E__inference_reshape_layer_call_and_return_conditional_losses_28848331

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
?e
?
G__inference_generator_layer_call_and_return_conditional_losses_28848023
input_18
$dense_matmul_readvariableop_resource:
??b4
%dense_biasadd_readvariableop_resource:	?bA
%conv2d_conv2d_readvariableop_resource:??5
&conv2d_biasadd_readvariableop_resource:	?A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinput_1#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense/BiasAdd?
glu/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu/strided_slice/stack?
glu/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu/strided_slice/stack_1?
glu/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu/strided_slice/stack_2?
glu/strided_sliceStridedSlicedense/BiasAdd:output:0 glu/strided_slice/stack:output:0"glu/strided_slice/stack_1:output:0"glu/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu/strided_slice?
glu/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu/strided_slice_1/stack?
glu/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu/strided_slice_1/stack_1?
glu/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu/strided_slice_1/stack_2?
glu/strided_slice_1StridedSlicedense/BiasAdd:output:0"glu/strided_slice_1/stack:output:0$glu/strided_slice_1/stack_1:output:0$glu/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu/strided_slice_1v
glu/SigmoidSigmoidglu/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu/Sigmoidy
glu/mulMulglu/strided_slice:output:0glu/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12	
glu/mulY
reshape/ShapeShapeglu/mul:z:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeglu/mul:z:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshape{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborreshape/Reshape:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAdd?
glu/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_2/stack?
glu/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu/strided_slice_2/stack_1?
glu/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_2/stack_2?
glu/strided_slice_2StridedSliceconv2d/BiasAdd:output:0"glu/strided_slice_2/stack:output:0$glu/strided_slice_2/stack_1:output:0$glu/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu/strided_slice_2?
glu/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu/strided_slice_3/stack?
glu/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_3/stack_1?
glu/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_3/stack_2?
glu/strided_slice_3StridedSliceconv2d/BiasAdd:output:0"glu/strided_slice_3/stack:output:0$glu/strided_slice_3/stack_1:output:0$glu/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu/strided_slice_3?
glu/Sigmoid_1Sigmoidglu/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu/Sigmoid_1?
	glu/mul_1Mulglu/strided_slice_2:output:0glu/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
	glu/mul_1
up_sampling2d/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_2
up_sampling2d/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_3?
up_sampling2d/mul_1Mulup_sampling2d/Const_2:output:0up_sampling2d/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul_1?
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborglu/mul_1:z:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,up_sampling2d/resize_1/ResizeNearestNeighbor?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
glu/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_4/stack?
glu/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_4/stack_1?
glu/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_4/stack_2?
glu/strided_slice_4StridedSliceconv2d_1/BiasAdd:output:0"glu/strided_slice_4/stack:output:0$glu/strided_slice_4/stack_1:output:0$glu/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu/strided_slice_4?
glu/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_5/stack?
glu/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_5/stack_1?
glu/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_5/stack_2?
glu/strided_slice_5StridedSliceconv2d_1/BiasAdd:output:0"glu/strided_slice_5/stack:output:0$glu/strided_slice_5/stack_1:output:0$glu/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu/strided_slice_5?
glu/Sigmoid_2Sigmoidglu/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu/Sigmoid_2?
	glu/mul_2Mulglu/strided_slice_4:output:0glu/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
	glu/mul_2?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dglu/mul_2:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/TanhTanhconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Tanh?
IdentityIdentityconv2d_2/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
)__inference_conv2d_layer_call_fn_28848355

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
GPU2*0,1J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_288474322
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
?	
?
,__inference_generator_layer_call_fn_28848184
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
GPU2*0,1J 8? *P
fKRI
G__inference_generator_layer_call_and_return_conditional_losses_288476692
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
?%
?
$__inference__traced_restore_28848475
file_prefix1
assignvariableop_dense_kernel:
??b,
assignvariableop_1_dense_bias:	?b<
 assignvariableop_2_conv2d_kernel:??-
assignvariableop_3_conv2d_bias:	?<
"assignvariableop_4_conv2d_1_kernel:@@.
 assignvariableop_5_conv2d_1_bias:@<
"assignvariableop_6_conv2d_2_kernel: .
 assignvariableop_7_conv2d_2_bias:

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
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_conv2d_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_conv2d_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_1_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_1_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*
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
?
]
A__inference_glu_layer_call_and_return_conditional_losses_28847483

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
B
&__inference_glu_layer_call_fn_28848317

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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288475872
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
?
g
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_28847375

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
?e
?
G__inference_generator_layer_call_and_return_conditional_losses_28847946

inputs8
$dense_matmul_readvariableop_resource:
??b4
%dense_biasadd_readvariableop_resource:	?bA
%conv2d_conv2d_readvariableop_resource:??5
&conv2d_biasadd_readvariableop_resource:	?A
'conv2d_1_conv2d_readvariableop_resource:@@6
(conv2d_1_biasadd_readvariableop_resource:@A
'conv2d_2_conv2d_readvariableop_resource: 6
(conv2d_2_biasadd_readvariableop_resource:
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense/BiasAdd?
glu/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu/strided_slice/stack?
glu/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu/strided_slice/stack_1?
glu/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu/strided_slice/stack_2?
glu/strided_sliceStridedSlicedense/BiasAdd:output:0 glu/strided_slice/stack:output:0"glu/strided_slice/stack_1:output:0"glu/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu/strided_slice?
glu/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu/strided_slice_1/stack?
glu/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu/strided_slice_1/stack_1?
glu/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu/strided_slice_1/stack_2?
glu/strided_slice_1StridedSlicedense/BiasAdd:output:0"glu/strided_slice_1/stack:output:0$glu/strided_slice_1/stack_1:output:0$glu/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu/strided_slice_1v
glu/SigmoidSigmoidglu/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu/Sigmoidy
glu/mulMulglu/strided_slice:output:0glu/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12	
glu/mulY
reshape/ShapeShapeglu/mul:z:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2u
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapeglu/mul:z:0reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape/Reshape{
up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const
up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_1?
up_sampling2d/mulMulup_sampling2d/Const:output:0up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul?
*up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighborreshape/Reshape:output:0up_sampling2d/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2,
*up_sampling2d/resize/ResizeNearestNeighbor?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2D;up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAdd?
glu/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_2/stack?
glu/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu/strided_slice_2/stack_1?
glu/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_2/stack_2?
glu/strided_slice_2StridedSliceconv2d/BiasAdd:output:0"glu/strided_slice_2/stack:output:0$glu/strided_slice_2/stack_1:output:0$glu/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu/strided_slice_2?
glu/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu/strided_slice_3/stack?
glu/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_3/stack_1?
glu/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_3/stack_2?
glu/strided_slice_3StridedSliceconv2d/BiasAdd:output:0"glu/strided_slice_3/stack:output:0$glu/strided_slice_3/stack_1:output:0$glu/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu/strided_slice_3?
glu/Sigmoid_1Sigmoidglu/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu/Sigmoid_1?
	glu/mul_1Mulglu/strided_slice_2:output:0glu/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
	glu/mul_1
up_sampling2d/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_2
up_sampling2d/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d/Const_3?
up_sampling2d/mul_1Mulup_sampling2d/Const_2:output:0up_sampling2d/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d/mul_1?
,up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborglu/mul_1:z:0up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2.
,up_sampling2d/resize_1/ResizeNearestNeighbor?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2D=up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd?
glu/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_4/stack?
glu/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_4/stack_1?
glu/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_4/stack_2?
glu/strided_slice_4StridedSliceconv2d_1/BiasAdd:output:0"glu/strided_slice_4/stack:output:0$glu/strided_slice_4/stack_1:output:0$glu/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu/strided_slice_4?
glu/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_5/stack?
glu/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu/strided_slice_5/stack_1?
glu/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu/strided_slice_5/stack_2?
glu/strided_slice_5StridedSliceconv2d_1/BiasAdd:output:0"glu/strided_slice_5/stack:output:0$glu/strided_slice_5/stack_1:output:0$glu/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu/strided_slice_5?
glu/Sigmoid_2Sigmoidglu/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu/Sigmoid_2?
	glu/mul_2Mulglu/strided_slice_4:output:0glu/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
	glu/mul_2?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dglu/mul_2:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_2/BiasAdd{
conv2d_2/TanhTanhconv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_2/Tanh?
IdentityIdentityconv2d_2/Tanh:y:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_dense_layer_call_fn_28848203

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
GPU2*0,1J 8? *L
fGRE
C__inference_dense_layer_call_and_return_conditional_losses_288473982
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
?
?
+__inference_conv2d_1_layer_call_fn_28848374

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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_288474642
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
?
]
A__inference_glu_layer_call_and_return_conditional_losses_28847302

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
?
]
A__inference_glu_layer_call_and_return_conditional_losses_28847587

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
a
E__inference_reshape_layer_call_and_return_conditional_losses_28847419

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
?
B
&__inference_glu_layer_call_fn_28848297

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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288473262
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
?
^
A__inference_glu_layer_call_and_return_conditional_losses_28847362
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
]
A__inference_glu_layer_call_and_return_conditional_losses_28848245

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
B
&__inference_glu_layer_call_fn_28848302

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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288474832
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
?
C
&__inference_glu_layer_call_fn_28847305
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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288473022
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
?x
?
#__inference__wrapped_model_28847284
input_1B
.generator_dense_matmul_readvariableop_resource:
??b>
/generator_dense_biasadd_readvariableop_resource:	?bK
/generator_conv2d_conv2d_readvariableop_resource:???
0generator_conv2d_biasadd_readvariableop_resource:	?K
1generator_conv2d_1_conv2d_readvariableop_resource:@@@
2generator_conv2d_1_biasadd_readvariableop_resource:@K
1generator_conv2d_2_conv2d_readvariableop_resource: @
2generator_conv2d_2_biasadd_readvariableop_resource:
identity??'generator/conv2d/BiasAdd/ReadVariableOp?&generator/conv2d/Conv2D/ReadVariableOp?)generator/conv2d_1/BiasAdd/ReadVariableOp?(generator/conv2d_1/Conv2D/ReadVariableOp?)generator/conv2d_2/BiasAdd/ReadVariableOp?(generator/conv2d_2/Conv2D/ReadVariableOp?&generator/dense/BiasAdd/ReadVariableOp?%generator/dense/MatMul/ReadVariableOp?
%generator/dense/MatMul/ReadVariableOpReadVariableOp.generator_dense_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02'
%generator/dense/MatMul/ReadVariableOp?
generator/dense/MatMulMatMulinput_1-generator/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
generator/dense/MatMul?
&generator/dense/BiasAdd/ReadVariableOpReadVariableOp/generator_dense_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02(
&generator/dense/BiasAdd/ReadVariableOp?
generator/dense/BiasAddBiasAdd generator/dense/MatMul:product:0.generator/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
generator/dense/BiasAdd?
!generator/glu/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2#
!generator/glu/strided_slice/stack?
#generator/glu/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#generator/glu/strided_slice/stack_1?
#generator/glu/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator/glu/strided_slice/stack_2?
generator/glu/strided_sliceStridedSlice generator/dense/BiasAdd:output:0*generator/glu/strided_slice/stack:output:0,generator/glu/strided_slice/stack_1:output:0,generator/glu/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
generator/glu/strided_slice?
#generator/glu/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2%
#generator/glu/strided_slice_1/stack?
%generator/glu/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2'
%generator/glu/strided_slice_1/stack_1?
%generator/glu/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2'
%generator/glu/strided_slice_1/stack_2?
generator/glu/strided_slice_1StridedSlice generator/dense/BiasAdd:output:0,generator/glu/strided_slice_1/stack:output:0.generator/glu/strided_slice_1/stack_1:output:0.generator/glu/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
generator/glu/strided_slice_1?
generator/glu/SigmoidSigmoid&generator/glu/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
generator/glu/Sigmoid?
generator/glu/mulMul$generator/glu/strided_slice:output:0generator/glu/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
generator/glu/mulw
generator/reshape/ShapeShapegenerator/glu/mul:z:0*
T0*
_output_shapes
:2
generator/reshape/Shape?
%generator/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2'
%generator/reshape/strided_slice/stack?
'generator/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2)
'generator/reshape/strided_slice/stack_1?
'generator/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2)
'generator/reshape/strided_slice/stack_2?
generator/reshape/strided_sliceStridedSlice generator/reshape/Shape:output:0.generator/reshape/strided_slice/stack:output:00generator/reshape/strided_slice/stack_1:output:00generator/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2!
generator/reshape/strided_slice?
!generator/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2#
!generator/reshape/Reshape/shape/1?
!generator/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2#
!generator/reshape/Reshape/shape/2?
!generator/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2#
!generator/reshape/Reshape/shape/3?
generator/reshape/Reshape/shapePack(generator/reshape/strided_slice:output:0*generator/reshape/Reshape/shape/1:output:0*generator/reshape/Reshape/shape/2:output:0*generator/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2!
generator/reshape/Reshape/shape?
generator/reshape/ReshapeReshapegenerator/glu/mul:z:0(generator/reshape/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
generator/reshape/Reshape?
generator/up_sampling2d/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
generator/up_sampling2d/Const?
generator/up_sampling2d/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2!
generator/up_sampling2d/Const_1?
generator/up_sampling2d/mulMul&generator/up_sampling2d/Const:output:0(generator/up_sampling2d/Const_1:output:0*
T0*
_output_shapes
:2
generator/up_sampling2d/mul?
4generator/up_sampling2d/resize/ResizeNearestNeighborResizeNearestNeighbor"generator/reshape/Reshape:output:0generator/up_sampling2d/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(26
4generator/up_sampling2d/resize/ResizeNearestNeighbor?
&generator/conv2d/Conv2D/ReadVariableOpReadVariableOp/generator_conv2d_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02(
&generator/conv2d/Conv2D/ReadVariableOp?
generator/conv2d/Conv2DConv2DEgenerator/up_sampling2d/resize/ResizeNearestNeighbor:resized_images:0.generator/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
generator/conv2d/Conv2D?
'generator/conv2d/BiasAdd/ReadVariableOpReadVariableOp0generator_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'generator/conv2d/BiasAdd/ReadVariableOp?
generator/conv2d/BiasAddBiasAdd generator/conv2d/Conv2D:output:0/generator/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
generator/conv2d/BiasAdd?
#generator/glu/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2%
#generator/glu/strided_slice_2/stack?
%generator/glu/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2'
%generator/glu/strided_slice_2/stack_1?
%generator/glu/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2'
%generator/glu/strided_slice_2/stack_2?
generator/glu/strided_slice_2StridedSlice!generator/conv2d/BiasAdd:output:0,generator/glu/strided_slice_2/stack:output:0.generator/glu/strided_slice_2/stack_1:output:0.generator/glu/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
generator/glu/strided_slice_2?
#generator/glu/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2%
#generator/glu/strided_slice_3/stack?
%generator/glu/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2'
%generator/glu/strided_slice_3/stack_1?
%generator/glu/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2'
%generator/glu/strided_slice_3/stack_2?
generator/glu/strided_slice_3StridedSlice!generator/conv2d/BiasAdd:output:0,generator/glu/strided_slice_3/stack:output:0.generator/glu/strided_slice_3/stack_1:output:0.generator/glu/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
generator/glu/strided_slice_3?
generator/glu/Sigmoid_1Sigmoid&generator/glu/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
generator/glu/Sigmoid_1?
generator/glu/mul_1Mul&generator/glu/strided_slice_2:output:0generator/glu/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
generator/glu/mul_1?
generator/up_sampling2d/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2!
generator/up_sampling2d/Const_2?
generator/up_sampling2d/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2!
generator/up_sampling2d/Const_3?
generator/up_sampling2d/mul_1Mul(generator/up_sampling2d/Const_2:output:0(generator/up_sampling2d/Const_3:output:0*
T0*
_output_shapes
:2
generator/up_sampling2d/mul_1?
6generator/up_sampling2d/resize_1/ResizeNearestNeighborResizeNearestNeighborgenerator/glu/mul_1:z:0!generator/up_sampling2d/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(28
6generator/up_sampling2d/resize_1/ResizeNearestNeighbor?
(generator/conv2d_1/Conv2D/ReadVariableOpReadVariableOp1generator_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(generator/conv2d_1/Conv2D/ReadVariableOp?
generator/conv2d_1/Conv2DConv2DGgenerator/up_sampling2d/resize_1/ResizeNearestNeighbor:resized_images:00generator/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
generator/conv2d_1/Conv2D?
)generator/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp2generator_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)generator/conv2d_1/BiasAdd/ReadVariableOp?
generator/conv2d_1/BiasAddBiasAdd"generator/conv2d_1/Conv2D:output:01generator/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
generator/conv2d_1/BiasAdd?
#generator/glu/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2%
#generator/glu/strided_slice_4/stack?
%generator/glu/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2'
%generator/glu/strided_slice_4/stack_1?
%generator/glu/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2'
%generator/glu/strided_slice_4/stack_2?
generator/glu/strided_slice_4StridedSlice#generator/conv2d_1/BiasAdd:output:0,generator/glu/strided_slice_4/stack:output:0.generator/glu/strided_slice_4/stack_1:output:0.generator/glu/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
generator/glu/strided_slice_4?
#generator/glu/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2%
#generator/glu/strided_slice_5/stack?
%generator/glu/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2'
%generator/glu/strided_slice_5/stack_1?
%generator/glu/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2'
%generator/glu/strided_slice_5/stack_2?
generator/glu/strided_slice_5StridedSlice#generator/conv2d_1/BiasAdd:output:0,generator/glu/strided_slice_5/stack:output:0.generator/glu/strided_slice_5/stack_1:output:0.generator/glu/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
generator/glu/strided_slice_5?
generator/glu/Sigmoid_2Sigmoid&generator/glu/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
generator/glu/Sigmoid_2?
generator/glu/mul_2Mul&generator/glu/strided_slice_4:output:0generator/glu/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
generator/glu/mul_2?
(generator/conv2d_2/Conv2D/ReadVariableOpReadVariableOp1generator_conv2d_2_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(generator/conv2d_2/Conv2D/ReadVariableOp?
generator/conv2d_2/Conv2DConv2Dgenerator/glu/mul_2:z:00generator/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
generator/conv2d_2/Conv2D?
)generator/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp2generator_conv2d_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)generator/conv2d_2/BiasAdd/ReadVariableOp?
generator/conv2d_2/BiasAddBiasAdd"generator/conv2d_2/Conv2D:output:01generator/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
generator/conv2d_2/BiasAdd?
generator/conv2d_2/TanhTanh#generator/conv2d_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
generator/conv2d_2/Tanh?
IdentityIdentitygenerator/conv2d_2/Tanh:y:0(^generator/conv2d/BiasAdd/ReadVariableOp'^generator/conv2d/Conv2D/ReadVariableOp*^generator/conv2d_1/BiasAdd/ReadVariableOp)^generator/conv2d_1/Conv2D/ReadVariableOp*^generator/conv2d_2/BiasAdd/ReadVariableOp)^generator/conv2d_2/Conv2D/ReadVariableOp'^generator/dense/BiasAdd/ReadVariableOp&^generator/dense/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2R
'generator/conv2d/BiasAdd/ReadVariableOp'generator/conv2d/BiasAdd/ReadVariableOp2P
&generator/conv2d/Conv2D/ReadVariableOp&generator/conv2d/Conv2D/ReadVariableOp2V
)generator/conv2d_1/BiasAdd/ReadVariableOp)generator/conv2d_1/BiasAdd/ReadVariableOp2T
(generator/conv2d_1/Conv2D/ReadVariableOp(generator/conv2d_1/Conv2D/ReadVariableOp2V
)generator/conv2d_2/BiasAdd/ReadVariableOp)generator/conv2d_2/BiasAdd/ReadVariableOp2T
(generator/conv2d_2/Conv2D/ReadVariableOp(generator/conv2d_2/Conv2D/ReadVariableOp2P
&generator/dense/BiasAdd/ReadVariableOp&generator/dense/BiasAdd/ReadVariableOp2N
%generator/dense/MatMul/ReadVariableOp%generator/dense/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
D__inference_conv2d_layer_call_and_return_conditional_losses_28848346

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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_28847464

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
?
B
&__inference_glu_layer_call_fn_28848312

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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288474512
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
?
?
+__inference_conv2d_2_layer_call_fn_28848394

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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_288474962
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
?
]
A__inference_glu_layer_call_and_return_conditional_losses_28847451

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
?
B
&__inference_glu_layer_call_fn_28848292

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
GPU2*0,1J 8? *J
fERC
A__inference_glu_layer_call_and_return_conditional_losses_288473022
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
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_28848385

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
?
,__inference_generator_layer_call_fn_28848142

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
GPU2*0,1J 8? *P
fKRI
G__inference_generator_layer_call_and_return_conditional_losses_288475032
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
regularization_losses
		variables

trainable_variables
	keras_api

signatures
*Y&call_and_return_all_conditional_losses
Z_default_save_signature
[__call__"?
_tf_keras_model?{"name": "generator", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Generator", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [128, 128]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Generator"}}
?

kernel
bias
regularization_losses
	variables
trainable_variables
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
regularization_losses
	variables
trainable_variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__"?
_tf_keras_model?{"name": "glu", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "GLU", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 12544]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "GLU"}}
?
regularization_losses
	variables
trainable_variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 128]}}, "shared_object_id": 4}
?
regularization_losses
	variables
trainable_variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_layer?{"name": "up_sampling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 6}}
?


kernel
 bias
!regularization_losses
"	variables
#trainable_variables
$	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?	
_tf_keras_layer?	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 128]}}
?


%kernel
&bias
'regularization_losses
(	variables
)trainable_variables
*	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?	
_tf_keras_layer?	{"name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 64]}}
?


+kernel
,bias
-regularization_losses
.	variables
/trainable_variables
0	keras_api
*h&call_and_return_all_conditional_losses
i__call__"?	
_tf_keras_layer?	{"name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
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
1layer_regularization_losses
regularization_losses

2layers
3metrics
4layer_metrics
		variables

trainable_variables
5non_trainable_variables
[__call__
Z_default_save_signature
*Y&call_and_return_all_conditional_losses
&Y"call_and_return_conditional_losses"
_generic_user_object
,
jserving_default"
signature_map
 :
??b2dense/kernel
:?b2
dense/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
6layer_regularization_losses
regularization_losses

7layers
8metrics
9layer_metrics
	variables
trainable_variables
:non_trainable_variables
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
;layer_regularization_losses
regularization_losses

<layers
=metrics
>layer_metrics
	variables
trainable_variables
?non_trainable_variables
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
@layer_regularization_losses
regularization_losses

Alayers
Bmetrics
Clayer_metrics
	variables
trainable_variables
Dnon_trainable_variables
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
Elayer_regularization_losses
regularization_losses

Flayers
Gmetrics
Hlayer_metrics
	variables
trainable_variables
Inon_trainable_variables
c__call__
*b&call_and_return_all_conditional_losses
&b"call_and_return_conditional_losses"
_generic_user_object
):'??2conv2d/kernel
:?2conv2d/bias
 "
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
.
0
 1"
trackable_list_wrapper
?
Jlayer_regularization_losses
!regularization_losses

Klayers
Lmetrics
Mlayer_metrics
"	variables
#trainable_variables
Nnon_trainable_variables
e__call__
*d&call_and_return_all_conditional_losses
&d"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_1/kernel
:@2conv2d_1/bias
 "
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
.
%0
&1"
trackable_list_wrapper
?
Olayer_regularization_losses
'regularization_losses

Players
Qmetrics
Rlayer_metrics
(	variables
)trainable_variables
Snon_trainable_variables
g__call__
*f&call_and_return_all_conditional_losses
&f"call_and_return_conditional_losses"
_generic_user_object
):' 2conv2d_2/kernel
:2conv2d_2/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
Tlayer_regularization_losses
-regularization_losses

Ulayers
Vmetrics
Wlayer_metrics
.	variables
/trainable_variables
Xnon_trainable_variables
i__call__
*h&call_and_return_all_conditional_losses
&h"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
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
 "
trackable_list_wrapper
?2?
G__inference_generator_layer_call_and_return_conditional_losses_28847869
G__inference_generator_layer_call_and_return_conditional_losses_28847946
G__inference_generator_layer_call_and_return_conditional_losses_28848023
G__inference_generator_layer_call_and_return_conditional_losses_28848100?
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
#__inference__wrapped_model_28847284?
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
?2?
,__inference_generator_layer_call_fn_28848121
,__inference_generator_layer_call_fn_28848142
,__inference_generator_layer_call_fn_28848163
,__inference_generator_layer_call_fn_28848184?
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
C__inference_dense_layer_call_and_return_conditional_losses_28848194?
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
(__inference_dense_layer_call_fn_28848203?
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
A__inference_glu_layer_call_and_return_conditional_losses_28848217
A__inference_glu_layer_call_and_return_conditional_losses_28848231
A__inference_glu_layer_call_and_return_conditional_losses_28847348
A__inference_glu_layer_call_and_return_conditional_losses_28847362
A__inference_glu_layer_call_and_return_conditional_losses_28848245
A__inference_glu_layer_call_and_return_conditional_losses_28848259
A__inference_glu_layer_call_and_return_conditional_losses_28848273
A__inference_glu_layer_call_and_return_conditional_losses_28848287?
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
&__inference_glu_layer_call_fn_28847305
&__inference_glu_layer_call_fn_28848292
&__inference_glu_layer_call_fn_28848297
&__inference_glu_layer_call_fn_28847334
&__inference_glu_layer_call_fn_28848302
&__inference_glu_layer_call_fn_28848307
&__inference_glu_layer_call_fn_28848312
&__inference_glu_layer_call_fn_28848317?
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
E__inference_reshape_layer_call_and_return_conditional_losses_28848331?
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
*__inference_reshape_layer_call_fn_28848336?
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
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_28847375?
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
0__inference_up_sampling2d_layer_call_fn_28847381?
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
D__inference_conv2d_layer_call_and_return_conditional_losses_28848346?
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
)__inference_conv2d_layer_call_fn_28848355?
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
F__inference_conv2d_1_layer_call_and_return_conditional_losses_28848365?
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
+__inference_conv2d_1_layer_call_fn_28848374?
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
F__inference_conv2d_2_layer_call_and_return_conditional_losses_28848385?
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
+__inference_conv2d_2_layer_call_fn_28848394?
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
&__inference_signature_wrapper_28847792input_1"?
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
#__inference__wrapped_model_28847284z %&+,1?.
'?$
"?
input_1??????????
? ";?8
6
output_1*?'
output_1??????????
F__inference_conv2d_1_layer_call_and_return_conditional_losses_28848365?%&I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
+__inference_conv2d_1_layer_call_fn_28848374?%&I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_28848385?+,I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
+__inference_conv2d_2_layer_call_fn_28848394?+,I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
D__inference_conv2d_layer_call_and_return_conditional_losses_28848346? J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
)__inference_conv2d_layer_call_fn_28848355? J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
C__inference_dense_layer_call_and_return_conditional_losses_28848194^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????b
? }
(__inference_dense_layer_call_fn_28848203Q0?-
&?#
!?
inputs??????????
? "???????????b?
G__inference_generator_layer_call_and_return_conditional_losses_28847869o %&+,4?1
*?'
!?
inputs??????????
p 
? "-?*
#? 
0?????????
? ?
G__inference_generator_layer_call_and_return_conditional_losses_28847946o %&+,4?1
*?'
!?
inputs??????????
p
? "-?*
#? 
0?????????
? ?
G__inference_generator_layer_call_and_return_conditional_losses_28848023p %&+,5?2
+?(
"?
input_1??????????
p 
? "-?*
#? 
0?????????
? ?
G__inference_generator_layer_call_and_return_conditional_losses_28848100p %&+,5?2
+?(
"?
input_1??????????
p
? "-?*
#? 
0?????????
? ?
,__inference_generator_layer_call_fn_28848121u %&+,5?2
+?(
"?
input_1??????????
p 
? "2?/+????????????????????????????
,__inference_generator_layer_call_fn_28848142t %&+,4?1
*?'
!?
inputs??????????
p 
? "2?/+????????????????????????????
,__inference_generator_layer_call_fn_28848163t %&+,4?1
*?'
!?
inputs??????????
p
? "2?/+????????????????????????????
,__inference_generator_layer_call_fn_28848184u %&+,5?2
+?(
"?
input_1??????????
p
? "2?/+????????????????????????????
A__inference_glu_layer_call_and_return_conditional_losses_28847348kA?>
'?$
"?
input_1??????????b
?

trainingp "&?#
?
0??????????1
? ?
A__inference_glu_layer_call_and_return_conditional_losses_28847362kA?>
'?$
"?
input_1??????????b
?

trainingp"&?#
?
0??????????1
? ?
A__inference_glu_layer_call_and_return_conditional_losses_28848217j@?=
&?#
!?
inputs??????????b
?

trainingp "&?#
?
0??????????1
? ?
A__inference_glu_layer_call_and_return_conditional_losses_28848231j@?=
&?#
!?
inputs??????????b
?

trainingp"&?#
?
0??????????1
? ?
A__inference_glu_layer_call_and_return_conditional_losses_28848245?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp "??<
5?2
0+???????????????????????????@
? ?
A__inference_glu_layer_call_and_return_conditional_losses_28848259?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp "??<
5?2
0+??????????????????????????? 
? ?
A__inference_glu_layer_call_and_return_conditional_losses_28848273?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp"??<
5?2
0+??????????????????????????? 
? ?
A__inference_glu_layer_call_and_return_conditional_losses_28848287?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp"??<
5?2
0+???????????????????????????@
? ?
&__inference_glu_layer_call_fn_28847305^A?>
'?$
"?
input_1??????????b
?

trainingp "???????????1?
&__inference_glu_layer_call_fn_28847334^A?>
'?$
"?
input_1??????????b
?

trainingp"???????????1?
&__inference_glu_layer_call_fn_28848292]@?=
&?#
!?
inputs??????????b
?

trainingp "???????????1?
&__inference_glu_layer_call_fn_28848297]@?=
&?#
!?
inputs??????????b
?

trainingp"???????????1?
&__inference_glu_layer_call_fn_28848302?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp "2?/+??????????????????????????? ?
&__inference_glu_layer_call_fn_28848307?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp"2?/+??????????????????????????? ?
&__inference_glu_layer_call_fn_28848312?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp "2?/+???????????????????????????@?
&__inference_glu_layer_call_fn_28848317?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp"2?/+???????????????????????????@?
E__inference_reshape_layer_call_and_return_conditional_losses_28848331b0?-
&?#
!?
inputs??????????1
? ".?+
$?!
0??????????
? ?
*__inference_reshape_layer_call_fn_28848336U0?-
&?#
!?
inputs??????????1
? "!????????????
&__inference_signature_wrapper_28847792? %&+,<?9
? 
2?/
-
input_1"?
input_1??????????";?8
6
output_1*?'
output_1??????????
K__inference_up_sampling2d_layer_call_and_return_conditional_losses_28847375?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
0__inference_up_sampling2d_layer_call_fn_28847381?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????