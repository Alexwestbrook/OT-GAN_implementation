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
dense_8/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??b*
shared_namedense_8/kernel
s
"dense_8/kernel/Read/ReadVariableOpReadVariableOpdense_8/kernel* 
_output_shapes
:
??b*
dtype0
q
dense_8/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?b*
shared_namedense_8/bias
j
 dense_8/bias/Read/ReadVariableOpReadVariableOpdense_8/bias*
_output_shapes	
:?b*
dtype0
?
conv2d_48/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_48/kernel

$conv2d_48/kernel/Read/ReadVariableOpReadVariableOpconv2d_48/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_48/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_48/bias
n
"conv2d_48/bias/Read/ReadVariableOpReadVariableOpconv2d_48/bias*
_output_shapes	
:?*
dtype0
?
conv2d_49/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_49/kernel
}
$conv2d_49/kernel/Read/ReadVariableOpReadVariableOpconv2d_49/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_49/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_49/bias
m
"conv2d_49/bias/Read/ReadVariableOpReadVariableOpconv2d_49/bias*
_output_shapes
:@*
dtype0
?
conv2d_50/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_50/kernel
}
$conv2d_50/kernel/Read/ReadVariableOpReadVariableOpconv2d_50/kernel*&
_output_shapes
: *
dtype0
t
conv2d_50/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_50/bias
m
"conv2d_50/bias/Read/ReadVariableOpReadVariableOpconv2d_50/bias*
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
VARIABLE_VALUEdense_8/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEdense_8/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE
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
MK
VARIABLE_VALUEconv2d_48/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_48/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
MK
VARIABLE_VALUEconv2d_49/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_49/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
MK
VARIABLE_VALUEconv2d_50/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_50/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_8/kerneldense_8/biasconv2d_48/kernelconv2d_48/biasconv2d_49/kernelconv2d_49/biasconv2d_50/kernelconv2d_50/bias*
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
GPU2*0,1J 8? *0
f+R)
'__inference_signature_wrapper_147418058
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_8/kernel/Read/ReadVariableOp dense_8/bias/Read/ReadVariableOp$conv2d_48/kernel/Read/ReadVariableOp"conv2d_48/bias/Read/ReadVariableOp$conv2d_49/kernel/Read/ReadVariableOp"conv2d_49/bias/Read/ReadVariableOp$conv2d_50/kernel/Read/ReadVariableOp"conv2d_50/bias/Read/ReadVariableOpConst*
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
GPU2*0,1J 8? *+
f&R$
"__inference__traced_save_147418707
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_8/kerneldense_8/biasconv2d_48/kernelconv2d_48/biasconv2d_49/kernelconv2d_49/biasconv2d_50/kernelconv2d_50/bias*
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
GPU2*0,1J 8? *.
f)R'
%__inference__traced_restore_147418741??	
?
I
-__inference_reshape_8_layer_call_fn_147418602

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
GPU2*0,1J 8? *Q
fLRJ
H__inference_reshape_8_layer_call_and_return_conditional_losses_1474176852
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
?
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147417568

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
H__inference_conv2d_50_layer_call_and_return_conditional_losses_147418651

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
?
E
)__inference_glu_8_layer_call_fn_147418558

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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474175682
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
?
"__inference__traced_save_147418707
file_prefix-
)savev2_dense_8_kernel_read_readvariableop+
'savev2_dense_8_bias_read_readvariableop/
+savev2_conv2d_48_kernel_read_readvariableop-
)savev2_conv2d_48_bias_read_readvariableop/
+savev2_conv2d_49_kernel_read_readvariableop-
)savev2_conv2d_49_bias_read_readvariableop/
+savev2_conv2d_50_kernel_read_readvariableop-
)savev2_conv2d_50_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_8_kernel_read_readvariableop'savev2_dense_8_bias_read_readvariableop+savev2_conv2d_48_kernel_read_readvariableop)savev2_conv2d_48_bias_read_readvariableop+savev2_conv2d_49_kernel_read_readvariableop)savev2_conv2d_49_bias_read_readvariableop+savev2_conv2d_50_kernel_read_readvariableop)savev2_conv2d_50_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147417853

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
?h
?
J__inference_generator_8_layer_call_and_return_conditional_losses_147418366
input_1:
&dense_8_matmul_readvariableop_resource:
??b6
'dense_8_biasadd_readvariableop_resource:	?bD
(conv2d_48_conv2d_readvariableop_resource:??8
)conv2d_48_biasadd_readvariableop_resource:	?B
(conv2d_49_conv2d_readvariableop_resource:@@7
)conv2d_49_biasadd_readvariableop_resource:@B
(conv2d_50_conv2d_readvariableop_resource: 7
)conv2d_50_biasadd_readvariableop_resource:
identity?? conv2d_48/BiasAdd/ReadVariableOp?conv2d_48/Conv2D/ReadVariableOp? conv2d_49/BiasAdd/ReadVariableOp?conv2d_49/Conv2D/ReadVariableOp? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulinput_1%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_8/BiasAdd?
glu_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_8/strided_slice/stack?
glu_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_8/strided_slice/stack_1?
glu_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_8/strided_slice/stack_2?
glu_8/strided_sliceStridedSlicedense_8/BiasAdd:output:0"glu_8/strided_slice/stack:output:0$glu_8/strided_slice/stack_1:output:0$glu_8/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_8/strided_slice?
glu_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_8/strided_slice_1/stack?
glu_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_8/strided_slice_1/stack_1?
glu_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_8/strided_slice_1/stack_2?
glu_8/strided_slice_1StridedSlicedense_8/BiasAdd:output:0$glu_8/strided_slice_1/stack:output:0&glu_8/strided_slice_1/stack_1:output:0&glu_8/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_8/strided_slice_1|
glu_8/SigmoidSigmoidglu_8/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_8/Sigmoid?
	glu_8/mulMulglu_8/strided_slice:output:0glu_8/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_8/mul_
reshape_8/ShapeShapeglu_8/mul:z:0*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2y
reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_8/Reshape/shape/3?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0"reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapeglu_8/mul:z:0 reshape_8/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_8/Reshape
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const?
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_1?
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_8/mul?
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborreshape_8/Reshape:output:0up_sampling2d_8/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_8/resize/ResizeNearestNeighbor?
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_48/Conv2D/ReadVariableOp?
conv2d_48/Conv2DConv2D=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_48/Conv2D?
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp?
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_48/BiasAdd?
glu_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_2/stack?
glu_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_8/strided_slice_2/stack_1?
glu_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_2/stack_2?
glu_8/strided_slice_2StridedSliceconv2d_48/BiasAdd:output:0$glu_8/strided_slice_2/stack:output:0&glu_8/strided_slice_2/stack_1:output:0&glu_8/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_8/strided_slice_2?
glu_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_8/strided_slice_3/stack?
glu_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_3/stack_1?
glu_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_3/stack_2?
glu_8/strided_slice_3StridedSliceconv2d_48/BiasAdd:output:0$glu_8/strided_slice_3/stack:output:0&glu_8/strided_slice_3/stack_1:output:0&glu_8/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_8/strided_slice_3?
glu_8/Sigmoid_1Sigmoidglu_8/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_8/Sigmoid_1?
glu_8/mul_1Mulglu_8/strided_slice_2:output:0glu_8/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_8/mul_1?
up_sampling2d_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_2?
up_sampling2d_8/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_3?
up_sampling2d_8/mul_1Mul up_sampling2d_8/Const_2:output:0 up_sampling2d_8/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_8/mul_1?
.up_sampling2d_8/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_8/mul_1:z:0up_sampling2d_8/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_8/resize_1/ResizeNearestNeighbor?
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_49/Conv2D/ReadVariableOp?
conv2d_49/Conv2DConv2D?up_sampling2d_8/resize_1/ResizeNearestNeighbor:resized_images:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_49/Conv2D?
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp?
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_49/BiasAdd?
glu_8/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_4/stack?
glu_8/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_4/stack_1?
glu_8/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_4/stack_2?
glu_8/strided_slice_4StridedSliceconv2d_49/BiasAdd:output:0$glu_8/strided_slice_4/stack:output:0&glu_8/strided_slice_4/stack_1:output:0&glu_8/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_8/strided_slice_4?
glu_8/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_5/stack?
glu_8/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_5/stack_1?
glu_8/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_5/stack_2?
glu_8/strided_slice_5StridedSliceconv2d_49/BiasAdd:output:0$glu_8/strided_slice_5/stack:output:0&glu_8/strided_slice_5/stack_1:output:0&glu_8/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_8/strided_slice_5?
glu_8/Sigmoid_2Sigmoidglu_8/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_8/Sigmoid_2?
glu_8/mul_2Mulglu_8/strided_slice_4:output:0glu_8/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_8/mul_2?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Dglu_8/mul_2:z:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_50/BiasAdd~
conv2d_50/TanhTanhconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_50/Tanh?
IdentityIdentityconv2d_50/Tanh:y:0!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
H__inference_conv2d_49_layer_call_and_return_conditional_losses_147417730

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
E
)__inference_glu_8_layer_call_fn_147418568

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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474177492
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
?
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147418497

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
H__inference_conv2d_49_layer_call_and_return_conditional_losses_147418631

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
?,
?
J__inference_generator_8_layer_call_and_return_conditional_losses_147417935

inputs%
dense_8_147417908:
??b 
dense_8_147417910:	?b/
conv2d_48_147417916:??"
conv2d_48_147417918:	?-
conv2d_49_147417923:@@!
conv2d_49_147417925:@-
conv2d_50_147417929: !
conv2d_50_147417931:
identity??!conv2d_48/StatefulPartitionedCall?!conv2d_49/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_147417908dense_8_147417910*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_1474176642!
dense_8/StatefulPartitionedCall?
glu_8/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474175922
glu_8/PartitionedCall?
reshape_8/PartitionedCallPartitionedCallglu_8/PartitionedCall:output:0*
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
GPU2*0,1J 8? *Q
fLRJ
H__inference_reshape_8_layer_call_and_return_conditional_losses_1474176852
reshape_8/PartitionedCall?
up_sampling2d_8/PartitionedCallPartitionedCall"reshape_8/PartitionedCall:output:0*
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
GPU2*0,1J 8? *W
fRRP
N__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1474176412!
up_sampling2d_8/PartitionedCall?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_8/PartitionedCall:output:0conv2d_48_147417916conv2d_48_147417918*
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
GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_48_layer_call_and_return_conditional_losses_1474176982#
!conv2d_48/StatefulPartitionedCall?
glu_8/PartitionedCall_1PartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474178532
glu_8/PartitionedCall_1?
!up_sampling2d_8/PartitionedCall_1PartitionedCall glu_8/PartitionedCall_1:output:0*
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
GPU2*0,1J 8? *W
fRRP
N__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1474176412#
!up_sampling2d_8/PartitionedCall_1?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall*up_sampling2d_8/PartitionedCall_1:output:0conv2d_49_147417923conv2d_49_147417925*
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
GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_49_layer_call_and_return_conditional_losses_1474177302#
!conv2d_49/StatefulPartitionedCall?
glu_8/PartitionedCall_2PartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474178192
glu_8/PartitionedCall_2?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall glu_8/PartitionedCall_2:output:0conv2d_50_147417929conv2d_50_147417931*
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
GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_50_layer_call_and_return_conditional_losses_1474177622#
!conv2d_50/StatefulPartitionedCall?
IdentityIdentity*conv2d_50/StatefulPartitionedCall:output:0"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147417819

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
d
H__inference_reshape_8_layer_call_and_return_conditional_losses_147418597

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
?h
?
J__inference_generator_8_layer_call_and_return_conditional_losses_147418212

inputs:
&dense_8_matmul_readvariableop_resource:
??b6
'dense_8_biasadd_readvariableop_resource:	?bD
(conv2d_48_conv2d_readvariableop_resource:??8
)conv2d_48_biasadd_readvariableop_resource:	?B
(conv2d_49_conv2d_readvariableop_resource:@@7
)conv2d_49_biasadd_readvariableop_resource:@B
(conv2d_50_conv2d_readvariableop_resource: 7
)conv2d_50_biasadd_readvariableop_resource:
identity?? conv2d_48/BiasAdd/ReadVariableOp?conv2d_48/Conv2D/ReadVariableOp? conv2d_49/BiasAdd/ReadVariableOp?conv2d_49/Conv2D/ReadVariableOp? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_8/BiasAdd?
glu_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_8/strided_slice/stack?
glu_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_8/strided_slice/stack_1?
glu_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_8/strided_slice/stack_2?
glu_8/strided_sliceStridedSlicedense_8/BiasAdd:output:0"glu_8/strided_slice/stack:output:0$glu_8/strided_slice/stack_1:output:0$glu_8/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_8/strided_slice?
glu_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_8/strided_slice_1/stack?
glu_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_8/strided_slice_1/stack_1?
glu_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_8/strided_slice_1/stack_2?
glu_8/strided_slice_1StridedSlicedense_8/BiasAdd:output:0$glu_8/strided_slice_1/stack:output:0&glu_8/strided_slice_1/stack_1:output:0&glu_8/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_8/strided_slice_1|
glu_8/SigmoidSigmoidglu_8/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_8/Sigmoid?
	glu_8/mulMulglu_8/strided_slice:output:0glu_8/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_8/mul_
reshape_8/ShapeShapeglu_8/mul:z:0*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2y
reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_8/Reshape/shape/3?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0"reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapeglu_8/mul:z:0 reshape_8/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_8/Reshape
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const?
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_1?
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_8/mul?
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborreshape_8/Reshape:output:0up_sampling2d_8/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_8/resize/ResizeNearestNeighbor?
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_48/Conv2D/ReadVariableOp?
conv2d_48/Conv2DConv2D=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_48/Conv2D?
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp?
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_48/BiasAdd?
glu_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_2/stack?
glu_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_8/strided_slice_2/stack_1?
glu_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_2/stack_2?
glu_8/strided_slice_2StridedSliceconv2d_48/BiasAdd:output:0$glu_8/strided_slice_2/stack:output:0&glu_8/strided_slice_2/stack_1:output:0&glu_8/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_8/strided_slice_2?
glu_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_8/strided_slice_3/stack?
glu_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_3/stack_1?
glu_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_3/stack_2?
glu_8/strided_slice_3StridedSliceconv2d_48/BiasAdd:output:0$glu_8/strided_slice_3/stack:output:0&glu_8/strided_slice_3/stack_1:output:0&glu_8/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_8/strided_slice_3?
glu_8/Sigmoid_1Sigmoidglu_8/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_8/Sigmoid_1?
glu_8/mul_1Mulglu_8/strided_slice_2:output:0glu_8/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_8/mul_1?
up_sampling2d_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_2?
up_sampling2d_8/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_3?
up_sampling2d_8/mul_1Mul up_sampling2d_8/Const_2:output:0 up_sampling2d_8/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_8/mul_1?
.up_sampling2d_8/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_8/mul_1:z:0up_sampling2d_8/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_8/resize_1/ResizeNearestNeighbor?
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_49/Conv2D/ReadVariableOp?
conv2d_49/Conv2DConv2D?up_sampling2d_8/resize_1/ResizeNearestNeighbor:resized_images:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_49/Conv2D?
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp?
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_49/BiasAdd?
glu_8/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_4/stack?
glu_8/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_4/stack_1?
glu_8/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_4/stack_2?
glu_8/strided_slice_4StridedSliceconv2d_49/BiasAdd:output:0$glu_8/strided_slice_4/stack:output:0&glu_8/strided_slice_4/stack_1:output:0&glu_8/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_8/strided_slice_4?
glu_8/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_5/stack?
glu_8/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_5/stack_1?
glu_8/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_5/stack_2?
glu_8/strided_slice_5StridedSliceconv2d_49/BiasAdd:output:0$glu_8/strided_slice_5/stack:output:0&glu_8/strided_slice_5/stack_1:output:0&glu_8/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_8/strided_slice_5?
glu_8/Sigmoid_2Sigmoidglu_8/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_8/Sigmoid_2?
glu_8/mul_2Mulglu_8/strided_slice_4:output:0glu_8/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_8/mul_2?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Dglu_8/mul_2:z:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_50/BiasAdd~
conv2d_50/TanhTanhconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_50/Tanh?
IdentityIdentityconv2d_50/Tanh:y:0!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_glu_8_layer_call_fn_147418578

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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474177172
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
-__inference_conv2d_48_layer_call_fn_147418621

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
GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_48_layer_call_and_return_conditional_losses_1474176982
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
H__inference_conv2d_48_layer_call_and_return_conditional_losses_147418612

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
?
?
-__inference_conv2d_49_layer_call_fn_147418640

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
GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_49_layer_call_and_return_conditional_losses_1474177302
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
?%
?
%__inference__traced_restore_147418741
file_prefix3
assignvariableop_dense_8_kernel:
??b.
assignvariableop_1_dense_8_bias:	?b?
#assignvariableop_2_conv2d_48_kernel:??0
!assignvariableop_3_conv2d_48_bias:	?=
#assignvariableop_4_conv2d_49_kernel:@@/
!assignvariableop_5_conv2d_49_bias:@=
#assignvariableop_6_conv2d_50_kernel: /
!assignvariableop_7_conv2d_50_bias:

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
AssignVariableOpAssignVariableOpassignvariableop_dense_8_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_8_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_48_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_48_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_49_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_49_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_50_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_50_biasIdentity_7:output:0"/device:CPU:0*
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
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147417717

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
F
)__inference_glu_8_layer_call_fn_147417571
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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474175682
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
?
/__inference_generator_8_layer_call_fn_147418429

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
GPU2*0,1J 8? *S
fNRL
J__inference_generator_8_layer_call_and_return_conditional_losses_1474179352
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
d
H__inference_reshape_8_layer_call_and_return_conditional_losses_147417685

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
?,
?
J__inference_generator_8_layer_call_and_return_conditional_losses_147417769

inputs%
dense_8_147417665:
??b 
dense_8_147417667:	?b/
conv2d_48_147417699:??"
conv2d_48_147417701:	?-
conv2d_49_147417731:@@!
conv2d_49_147417733:@-
conv2d_50_147417763: !
conv2d_50_147417765:
identity??!conv2d_48/StatefulPartitionedCall?!conv2d_49/StatefulPartitionedCall?!conv2d_50/StatefulPartitionedCall?dense_8/StatefulPartitionedCall?
dense_8/StatefulPartitionedCallStatefulPartitionedCallinputsdense_8_147417665dense_8_147417667*
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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_1474176642!
dense_8/StatefulPartitionedCall?
glu_8/PartitionedCallPartitionedCall(dense_8/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474175682
glu_8/PartitionedCall?
reshape_8/PartitionedCallPartitionedCallglu_8/PartitionedCall:output:0*
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
GPU2*0,1J 8? *Q
fLRJ
H__inference_reshape_8_layer_call_and_return_conditional_losses_1474176852
reshape_8/PartitionedCall?
up_sampling2d_8/PartitionedCallPartitionedCall"reshape_8/PartitionedCall:output:0*
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
GPU2*0,1J 8? *W
fRRP
N__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1474176412!
up_sampling2d_8/PartitionedCall?
!conv2d_48/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_8/PartitionedCall:output:0conv2d_48_147417699conv2d_48_147417701*
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
GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_48_layer_call_and_return_conditional_losses_1474176982#
!conv2d_48/StatefulPartitionedCall?
glu_8/PartitionedCall_1PartitionedCall*conv2d_48/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474177172
glu_8/PartitionedCall_1?
!up_sampling2d_8/PartitionedCall_1PartitionedCall glu_8/PartitionedCall_1:output:0*
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
GPU2*0,1J 8? *W
fRRP
N__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1474176412#
!up_sampling2d_8/PartitionedCall_1?
!conv2d_49/StatefulPartitionedCallStatefulPartitionedCall*up_sampling2d_8/PartitionedCall_1:output:0conv2d_49_147417731conv2d_49_147417733*
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
GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_49_layer_call_and_return_conditional_losses_1474177302#
!conv2d_49/StatefulPartitionedCall?
glu_8/PartitionedCall_2PartitionedCall*conv2d_49/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474177492
glu_8/PartitionedCall_2?
!conv2d_50/StatefulPartitionedCallStatefulPartitionedCall glu_8/PartitionedCall_2:output:0conv2d_50_147417763conv2d_50_147417765*
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
GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_50_layer_call_and_return_conditional_losses_1474177622#
!conv2d_50/StatefulPartitionedCall?
IdentityIdentity*conv2d_50/StatefulPartitionedCall:output:0"^conv2d_48/StatefulPartitionedCall"^conv2d_49/StatefulPartitionedCall"^conv2d_50/StatefulPartitionedCall ^dense_8/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2F
!conv2d_48/StatefulPartitionedCall!conv2d_48/StatefulPartitionedCall2F
!conv2d_49/StatefulPartitionedCall!conv2d_49/StatefulPartitionedCall2F
!conv2d_50/StatefulPartitionedCall!conv2d_50/StatefulPartitionedCall2B
dense_8/StatefulPartitionedCalldense_8/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
/__inference_generator_8_layer_call_fn_147418408

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
GPU2*0,1J 8? *S
fNRL
J__inference_generator_8_layer_call_and_return_conditional_losses_1474177692
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
?
F
)__inference_glu_8_layer_call_fn_147417600
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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474175922
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
E
)__inference_glu_8_layer_call_fn_147418583

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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474178532
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
j
N__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_147417641

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
?
?
+__inference_dense_8_layer_call_fn_147418469

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
GPU2*0,1J 8? *O
fJRH
F__inference_dense_8_layer_call_and_return_conditional_losses_1474176642
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
a
D__inference_glu_8_layer_call_and_return_conditional_losses_147417628
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
?h
?
J__inference_generator_8_layer_call_and_return_conditional_losses_147418289
input_1:
&dense_8_matmul_readvariableop_resource:
??b6
'dense_8_biasadd_readvariableop_resource:	?bD
(conv2d_48_conv2d_readvariableop_resource:??8
)conv2d_48_biasadd_readvariableop_resource:	?B
(conv2d_49_conv2d_readvariableop_resource:@@7
)conv2d_49_biasadd_readvariableop_resource:@B
(conv2d_50_conv2d_readvariableop_resource: 7
)conv2d_50_biasadd_readvariableop_resource:
identity?? conv2d_48/BiasAdd/ReadVariableOp?conv2d_48/Conv2D/ReadVariableOp? conv2d_49/BiasAdd/ReadVariableOp?conv2d_49/Conv2D/ReadVariableOp? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulinput_1%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_8/BiasAdd?
glu_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_8/strided_slice/stack?
glu_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_8/strided_slice/stack_1?
glu_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_8/strided_slice/stack_2?
glu_8/strided_sliceStridedSlicedense_8/BiasAdd:output:0"glu_8/strided_slice/stack:output:0$glu_8/strided_slice/stack_1:output:0$glu_8/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_8/strided_slice?
glu_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_8/strided_slice_1/stack?
glu_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_8/strided_slice_1/stack_1?
glu_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_8/strided_slice_1/stack_2?
glu_8/strided_slice_1StridedSlicedense_8/BiasAdd:output:0$glu_8/strided_slice_1/stack:output:0&glu_8/strided_slice_1/stack_1:output:0&glu_8/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_8/strided_slice_1|
glu_8/SigmoidSigmoidglu_8/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_8/Sigmoid?
	glu_8/mulMulglu_8/strided_slice:output:0glu_8/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_8/mul_
reshape_8/ShapeShapeglu_8/mul:z:0*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2y
reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_8/Reshape/shape/3?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0"reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapeglu_8/mul:z:0 reshape_8/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_8/Reshape
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const?
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_1?
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_8/mul?
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborreshape_8/Reshape:output:0up_sampling2d_8/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_8/resize/ResizeNearestNeighbor?
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_48/Conv2D/ReadVariableOp?
conv2d_48/Conv2DConv2D=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_48/Conv2D?
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp?
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_48/BiasAdd?
glu_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_2/stack?
glu_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_8/strided_slice_2/stack_1?
glu_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_2/stack_2?
glu_8/strided_slice_2StridedSliceconv2d_48/BiasAdd:output:0$glu_8/strided_slice_2/stack:output:0&glu_8/strided_slice_2/stack_1:output:0&glu_8/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_8/strided_slice_2?
glu_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_8/strided_slice_3/stack?
glu_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_3/stack_1?
glu_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_3/stack_2?
glu_8/strided_slice_3StridedSliceconv2d_48/BiasAdd:output:0$glu_8/strided_slice_3/stack:output:0&glu_8/strided_slice_3/stack_1:output:0&glu_8/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_8/strided_slice_3?
glu_8/Sigmoid_1Sigmoidglu_8/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_8/Sigmoid_1?
glu_8/mul_1Mulglu_8/strided_slice_2:output:0glu_8/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_8/mul_1?
up_sampling2d_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_2?
up_sampling2d_8/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_3?
up_sampling2d_8/mul_1Mul up_sampling2d_8/Const_2:output:0 up_sampling2d_8/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_8/mul_1?
.up_sampling2d_8/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_8/mul_1:z:0up_sampling2d_8/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_8/resize_1/ResizeNearestNeighbor?
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_49/Conv2D/ReadVariableOp?
conv2d_49/Conv2DConv2D?up_sampling2d_8/resize_1/ResizeNearestNeighbor:resized_images:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_49/Conv2D?
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp?
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_49/BiasAdd?
glu_8/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_4/stack?
glu_8/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_4/stack_1?
glu_8/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_4/stack_2?
glu_8/strided_slice_4StridedSliceconv2d_49/BiasAdd:output:0$glu_8/strided_slice_4/stack:output:0&glu_8/strided_slice_4/stack_1:output:0&glu_8/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_8/strided_slice_4?
glu_8/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_5/stack?
glu_8/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_5/stack_1?
glu_8/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_5/stack_2?
glu_8/strided_slice_5StridedSliceconv2d_49/BiasAdd:output:0$glu_8/strided_slice_5/stack:output:0&glu_8/strided_slice_5/stack_1:output:0&glu_8/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_8/strided_slice_5?
glu_8/Sigmoid_2Sigmoidglu_8/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_8/Sigmoid_2?
glu_8/mul_2Mulglu_8/strided_slice_4:output:0glu_8/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_8/mul_2?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Dglu_8/mul_2:z:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_50/BiasAdd~
conv2d_50/TanhTanhconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_50/Tanh?
IdentityIdentityconv2d_50/Tanh:y:0!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
F__inference_dense_8_layer_call_and_return_conditional_losses_147417664

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
H__inference_conv2d_50_layer_call_and_return_conditional_losses_147417762

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
?h
?
J__inference_generator_8_layer_call_and_return_conditional_losses_147418135

inputs:
&dense_8_matmul_readvariableop_resource:
??b6
'dense_8_biasadd_readvariableop_resource:	?bD
(conv2d_48_conv2d_readvariableop_resource:??8
)conv2d_48_biasadd_readvariableop_resource:	?B
(conv2d_49_conv2d_readvariableop_resource:@@7
)conv2d_49_biasadd_readvariableop_resource:@B
(conv2d_50_conv2d_readvariableop_resource: 7
)conv2d_50_biasadd_readvariableop_resource:
identity?? conv2d_48/BiasAdd/ReadVariableOp?conv2d_48/Conv2D/ReadVariableOp? conv2d_49/BiasAdd/ReadVariableOp?conv2d_49/Conv2D/ReadVariableOp? conv2d_50/BiasAdd/ReadVariableOp?conv2d_50/Conv2D/ReadVariableOp?dense_8/BiasAdd/ReadVariableOp?dense_8/MatMul/ReadVariableOp?
dense_8/MatMul/ReadVariableOpReadVariableOp&dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_8/MatMul/ReadVariableOp?
dense_8/MatMulMatMulinputs%dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_8/MatMul?
dense_8/BiasAdd/ReadVariableOpReadVariableOp'dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_8/BiasAdd/ReadVariableOp?
dense_8/BiasAddBiasAdddense_8/MatMul:product:0&dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_8/BiasAdd?
glu_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_8/strided_slice/stack?
glu_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_8/strided_slice/stack_1?
glu_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_8/strided_slice/stack_2?
glu_8/strided_sliceStridedSlicedense_8/BiasAdd:output:0"glu_8/strided_slice/stack:output:0$glu_8/strided_slice/stack_1:output:0$glu_8/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_8/strided_slice?
glu_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_8/strided_slice_1/stack?
glu_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_8/strided_slice_1/stack_1?
glu_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_8/strided_slice_1/stack_2?
glu_8/strided_slice_1StridedSlicedense_8/BiasAdd:output:0$glu_8/strided_slice_1/stack:output:0&glu_8/strided_slice_1/stack_1:output:0&glu_8/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_8/strided_slice_1|
glu_8/SigmoidSigmoidglu_8/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_8/Sigmoid?
	glu_8/mulMulglu_8/strided_slice:output:0glu_8/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_8/mul_
reshape_8/ShapeShapeglu_8/mul:z:0*
T0*
_output_shapes
:2
reshape_8/Shape?
reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_8/strided_slice/stack?
reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_1?
reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_8/strided_slice/stack_2?
reshape_8/strided_sliceStridedSlicereshape_8/Shape:output:0&reshape_8/strided_slice/stack:output:0(reshape_8/strided_slice/stack_1:output:0(reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_8/strided_slicex
reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/1x
reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_8/Reshape/shape/2y
reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_8/Reshape/shape/3?
reshape_8/Reshape/shapePack reshape_8/strided_slice:output:0"reshape_8/Reshape/shape/1:output:0"reshape_8/Reshape/shape/2:output:0"reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_8/Reshape/shape?
reshape_8/ReshapeReshapeglu_8/mul:z:0 reshape_8/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_8/Reshape
up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const?
up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_1?
up_sampling2d_8/mulMulup_sampling2d_8/Const:output:0 up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_8/mul?
,up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighborreshape_8/Reshape:output:0up_sampling2d_8/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_8/resize/ResizeNearestNeighbor?
conv2d_48/Conv2D/ReadVariableOpReadVariableOp(conv2d_48_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_48/Conv2D/ReadVariableOp?
conv2d_48/Conv2DConv2D=up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:0'conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_48/Conv2D?
 conv2d_48/BiasAdd/ReadVariableOpReadVariableOp)conv2d_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_48/BiasAdd/ReadVariableOp?
conv2d_48/BiasAddBiasAddconv2d_48/Conv2D:output:0(conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_48/BiasAdd?
glu_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_2/stack?
glu_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_8/strided_slice_2/stack_1?
glu_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_2/stack_2?
glu_8/strided_slice_2StridedSliceconv2d_48/BiasAdd:output:0$glu_8/strided_slice_2/stack:output:0&glu_8/strided_slice_2/stack_1:output:0&glu_8/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_8/strided_slice_2?
glu_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_8/strided_slice_3/stack?
glu_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_3/stack_1?
glu_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_3/stack_2?
glu_8/strided_slice_3StridedSliceconv2d_48/BiasAdd:output:0$glu_8/strided_slice_3/stack:output:0&glu_8/strided_slice_3/stack_1:output:0&glu_8/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_8/strided_slice_3?
glu_8/Sigmoid_1Sigmoidglu_8/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_8/Sigmoid_1?
glu_8/mul_1Mulglu_8/strided_slice_2:output:0glu_8/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_8/mul_1?
up_sampling2d_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_2?
up_sampling2d_8/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_8/Const_3?
up_sampling2d_8/mul_1Mul up_sampling2d_8/Const_2:output:0 up_sampling2d_8/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_8/mul_1?
.up_sampling2d_8/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_8/mul_1:z:0up_sampling2d_8/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_8/resize_1/ResizeNearestNeighbor?
conv2d_49/Conv2D/ReadVariableOpReadVariableOp(conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_49/Conv2D/ReadVariableOp?
conv2d_49/Conv2DConv2D?up_sampling2d_8/resize_1/ResizeNearestNeighbor:resized_images:0'conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_49/Conv2D?
 conv2d_49/BiasAdd/ReadVariableOpReadVariableOp)conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_49/BiasAdd/ReadVariableOp?
conv2d_49/BiasAddBiasAddconv2d_49/Conv2D:output:0(conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_49/BiasAdd?
glu_8/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_4/stack?
glu_8/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_4/stack_1?
glu_8/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_4/stack_2?
glu_8/strided_slice_4StridedSliceconv2d_49/BiasAdd:output:0$glu_8/strided_slice_4/stack:output:0&glu_8/strided_slice_4/stack_1:output:0&glu_8/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_8/strided_slice_4?
glu_8/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_5/stack?
glu_8/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_8/strided_slice_5/stack_1?
glu_8/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_8/strided_slice_5/stack_2?
glu_8/strided_slice_5StridedSliceconv2d_49/BiasAdd:output:0$glu_8/strided_slice_5/stack:output:0&glu_8/strided_slice_5/stack_1:output:0&glu_8/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_8/strided_slice_5?
glu_8/Sigmoid_2Sigmoidglu_8/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_8/Sigmoid_2?
glu_8/mul_2Mulglu_8/strided_slice_4:output:0glu_8/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_8/mul_2?
conv2d_50/Conv2D/ReadVariableOpReadVariableOp(conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_50/Conv2D/ReadVariableOp?
conv2d_50/Conv2DConv2Dglu_8/mul_2:z:0'conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_50/Conv2D?
 conv2d_50/BiasAdd/ReadVariableOpReadVariableOp)conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_50/BiasAdd/ReadVariableOp?
conv2d_50/BiasAddBiasAddconv2d_50/Conv2D:output:0(conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_50/BiasAdd~
conv2d_50/TanhTanhconv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_50/Tanh?
IdentityIdentityconv2d_50/Tanh:y:0!^conv2d_48/BiasAdd/ReadVariableOp ^conv2d_48/Conv2D/ReadVariableOp!^conv2d_49/BiasAdd/ReadVariableOp ^conv2d_49/Conv2D/ReadVariableOp!^conv2d_50/BiasAdd/ReadVariableOp ^conv2d_50/Conv2D/ReadVariableOp^dense_8/BiasAdd/ReadVariableOp^dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_48/BiasAdd/ReadVariableOp conv2d_48/BiasAdd/ReadVariableOp2B
conv2d_48/Conv2D/ReadVariableOpconv2d_48/Conv2D/ReadVariableOp2D
 conv2d_49/BiasAdd/ReadVariableOp conv2d_49/BiasAdd/ReadVariableOp2B
conv2d_49/Conv2D/ReadVariableOpconv2d_49/Conv2D/ReadVariableOp2D
 conv2d_50/BiasAdd/ReadVariableOp conv2d_50/BiasAdd/ReadVariableOp2B
conv2d_50/Conv2D/ReadVariableOpconv2d_50/Conv2D/ReadVariableOp2@
dense_8/BiasAdd/ReadVariableOpdense_8/BiasAdd/ReadVariableOp2>
dense_8/MatMul/ReadVariableOpdense_8/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
a
D__inference_glu_8_layer_call_and_return_conditional_losses_147417614
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
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147417749

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
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147418511

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
E
)__inference_glu_8_layer_call_fn_147418573

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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474178192
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
?
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147417592

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
F__inference_dense_8_layer_call_and_return_conditional_losses_147418460

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
?
?
-__inference_conv2d_50_layer_call_fn_147418660

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
GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_50_layer_call_and_return_conditional_losses_1474177622
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
?
H__inference_conv2d_48_layer_call_and_return_conditional_losses_147417698

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
?
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147418483

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
O
3__inference_up_sampling2d_8_layer_call_fn_147417647

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
GPU2*0,1J 8? *W
fRRP
N__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_1474176412
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
?	
?
/__inference_generator_8_layer_call_fn_147418387
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
GPU2*0,1J 8? *S
fNRL
J__inference_generator_8_layer_call_and_return_conditional_losses_1474177692
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
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147418525

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
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147418553

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
?
'__inference_signature_wrapper_147418058
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
GPU2*0,1J 8? *-
f(R&
$__inference__wrapped_model_1474175502
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
?	
?
/__inference_generator_8_layer_call_fn_147418450
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
GPU2*0,1J 8? *S
fNRL
J__inference_generator_8_layer_call_and_return_conditional_losses_1474179352
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
?
?
$__inference__wrapped_model_147417550
input_1F
2generator_8_dense_8_matmul_readvariableop_resource:
??bB
3generator_8_dense_8_biasadd_readvariableop_resource:	?bP
4generator_8_conv2d_48_conv2d_readvariableop_resource:??D
5generator_8_conv2d_48_biasadd_readvariableop_resource:	?N
4generator_8_conv2d_49_conv2d_readvariableop_resource:@@C
5generator_8_conv2d_49_biasadd_readvariableop_resource:@N
4generator_8_conv2d_50_conv2d_readvariableop_resource: C
5generator_8_conv2d_50_biasadd_readvariableop_resource:
identity??,generator_8/conv2d_48/BiasAdd/ReadVariableOp?+generator_8/conv2d_48/Conv2D/ReadVariableOp?,generator_8/conv2d_49/BiasAdd/ReadVariableOp?+generator_8/conv2d_49/Conv2D/ReadVariableOp?,generator_8/conv2d_50/BiasAdd/ReadVariableOp?+generator_8/conv2d_50/Conv2D/ReadVariableOp?*generator_8/dense_8/BiasAdd/ReadVariableOp?)generator_8/dense_8/MatMul/ReadVariableOp?
)generator_8/dense_8/MatMul/ReadVariableOpReadVariableOp2generator_8_dense_8_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02+
)generator_8/dense_8/MatMul/ReadVariableOp?
generator_8/dense_8/MatMulMatMulinput_11generator_8/dense_8/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
generator_8/dense_8/MatMul?
*generator_8/dense_8/BiasAdd/ReadVariableOpReadVariableOp3generator_8_dense_8_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02,
*generator_8/dense_8/BiasAdd/ReadVariableOp?
generator_8/dense_8/BiasAddBiasAdd$generator_8/dense_8/MatMul:product:02generator_8/dense_8/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
generator_8/dense_8/BiasAdd?
%generator_8/glu_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%generator_8/glu_8/strided_slice/stack?
'generator_8/glu_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'generator_8/glu_8/strided_slice/stack_1?
'generator_8/glu_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'generator_8/glu_8/strided_slice/stack_2?
generator_8/glu_8/strided_sliceStridedSlice$generator_8/dense_8/BiasAdd:output:0.generator_8/glu_8/strided_slice/stack:output:00generator_8/glu_8/strided_slice/stack_1:output:00generator_8/glu_8/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2!
generator_8/glu_8/strided_slice?
'generator_8/glu_8/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'generator_8/glu_8/strided_slice_1/stack?
)generator_8/glu_8/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)generator_8/glu_8/strided_slice_1/stack_1?
)generator_8/glu_8/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)generator_8/glu_8/strided_slice_1/stack_2?
!generator_8/glu_8/strided_slice_1StridedSlice$generator_8/dense_8/BiasAdd:output:00generator_8/glu_8/strided_slice_1/stack:output:02generator_8/glu_8/strided_slice_1/stack_1:output:02generator_8/glu_8/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2#
!generator_8/glu_8/strided_slice_1?
generator_8/glu_8/SigmoidSigmoid*generator_8/glu_8/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
generator_8/glu_8/Sigmoid?
generator_8/glu_8/mulMul(generator_8/glu_8/strided_slice:output:0generator_8/glu_8/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
generator_8/glu_8/mul?
generator_8/reshape_8/ShapeShapegenerator_8/glu_8/mul:z:0*
T0*
_output_shapes
:2
generator_8/reshape_8/Shape?
)generator_8/reshape_8/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)generator_8/reshape_8/strided_slice/stack?
+generator_8/reshape_8/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+generator_8/reshape_8/strided_slice/stack_1?
+generator_8/reshape_8/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+generator_8/reshape_8/strided_slice/stack_2?
#generator_8/reshape_8/strided_sliceStridedSlice$generator_8/reshape_8/Shape:output:02generator_8/reshape_8/strided_slice/stack:output:04generator_8/reshape_8/strided_slice/stack_1:output:04generator_8/reshape_8/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#generator_8/reshape_8/strided_slice?
%generator_8/reshape_8/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%generator_8/reshape_8/Reshape/shape/1?
%generator_8/reshape_8/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%generator_8/reshape_8/Reshape/shape/2?
%generator_8/reshape_8/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2'
%generator_8/reshape_8/Reshape/shape/3?
#generator_8/reshape_8/Reshape/shapePack,generator_8/reshape_8/strided_slice:output:0.generator_8/reshape_8/Reshape/shape/1:output:0.generator_8/reshape_8/Reshape/shape/2:output:0.generator_8/reshape_8/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#generator_8/reshape_8/Reshape/shape?
generator_8/reshape_8/ReshapeReshapegenerator_8/glu_8/mul:z:0,generator_8/reshape_8/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
generator_8/reshape_8/Reshape?
!generator_8/up_sampling2d_8/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!generator_8/up_sampling2d_8/Const?
#generator_8/up_sampling2d_8/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator_8/up_sampling2d_8/Const_1?
generator_8/up_sampling2d_8/mulMul*generator_8/up_sampling2d_8/Const:output:0,generator_8/up_sampling2d_8/Const_1:output:0*
T0*
_output_shapes
:2!
generator_8/up_sampling2d_8/mul?
8generator_8/up_sampling2d_8/resize/ResizeNearestNeighborResizeNearestNeighbor&generator_8/reshape_8/Reshape:output:0#generator_8/up_sampling2d_8/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2:
8generator_8/up_sampling2d_8/resize/ResizeNearestNeighbor?
+generator_8/conv2d_48/Conv2D/ReadVariableOpReadVariableOp4generator_8_conv2d_48_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+generator_8/conv2d_48/Conv2D/ReadVariableOp?
generator_8/conv2d_48/Conv2DConv2DIgenerator_8/up_sampling2d_8/resize/ResizeNearestNeighbor:resized_images:03generator_8/conv2d_48/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
generator_8/conv2d_48/Conv2D?
,generator_8/conv2d_48/BiasAdd/ReadVariableOpReadVariableOp5generator_8_conv2d_48_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,generator_8/conv2d_48/BiasAdd/ReadVariableOp?
generator_8/conv2d_48/BiasAddBiasAdd%generator_8/conv2d_48/Conv2D:output:04generator_8/conv2d_48/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
generator_8/conv2d_48/BiasAdd?
'generator_8/glu_8/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2)
'generator_8/glu_8/strided_slice_2/stack?
)generator_8/glu_8/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2+
)generator_8/glu_8/strided_slice_2/stack_1?
)generator_8/glu_8/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_8/glu_8/strided_slice_2/stack_2?
!generator_8/glu_8/strided_slice_2StridedSlice&generator_8/conv2d_48/BiasAdd:output:00generator_8/glu_8/strided_slice_2/stack:output:02generator_8/glu_8/strided_slice_2/stack_1:output:02generator_8/glu_8/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2#
!generator_8/glu_8/strided_slice_2?
'generator_8/glu_8/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2)
'generator_8/glu_8/strided_slice_3/stack?
)generator_8/glu_8/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2+
)generator_8/glu_8/strided_slice_3/stack_1?
)generator_8/glu_8/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_8/glu_8/strided_slice_3/stack_2?
!generator_8/glu_8/strided_slice_3StridedSlice&generator_8/conv2d_48/BiasAdd:output:00generator_8/glu_8/strided_slice_3/stack:output:02generator_8/glu_8/strided_slice_3/stack_1:output:02generator_8/glu_8/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2#
!generator_8/glu_8/strided_slice_3?
generator_8/glu_8/Sigmoid_1Sigmoid*generator_8/glu_8/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
generator_8/glu_8/Sigmoid_1?
generator_8/glu_8/mul_1Mul*generator_8/glu_8/strided_slice_2:output:0generator_8/glu_8/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
generator_8/glu_8/mul_1?
#generator_8/up_sampling2d_8/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator_8/up_sampling2d_8/Const_2?
#generator_8/up_sampling2d_8/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator_8/up_sampling2d_8/Const_3?
!generator_8/up_sampling2d_8/mul_1Mul,generator_8/up_sampling2d_8/Const_2:output:0,generator_8/up_sampling2d_8/Const_3:output:0*
T0*
_output_shapes
:2#
!generator_8/up_sampling2d_8/mul_1?
:generator_8/up_sampling2d_8/resize_1/ResizeNearestNeighborResizeNearestNeighborgenerator_8/glu_8/mul_1:z:0%generator_8/up_sampling2d_8/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2<
:generator_8/up_sampling2d_8/resize_1/ResizeNearestNeighbor?
+generator_8/conv2d_49/Conv2D/ReadVariableOpReadVariableOp4generator_8_conv2d_49_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+generator_8/conv2d_49/Conv2D/ReadVariableOp?
generator_8/conv2d_49/Conv2DConv2DKgenerator_8/up_sampling2d_8/resize_1/ResizeNearestNeighbor:resized_images:03generator_8/conv2d_49/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
generator_8/conv2d_49/Conv2D?
,generator_8/conv2d_49/BiasAdd/ReadVariableOpReadVariableOp5generator_8_conv2d_49_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,generator_8/conv2d_49/BiasAdd/ReadVariableOp?
generator_8/conv2d_49/BiasAddBiasAdd%generator_8/conv2d_49/Conv2D:output:04generator_8/conv2d_49/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
generator_8/conv2d_49/BiasAdd?
'generator_8/glu_8/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2)
'generator_8/glu_8/strided_slice_4/stack?
)generator_8/glu_8/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2+
)generator_8/glu_8/strided_slice_4/stack_1?
)generator_8/glu_8/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_8/glu_8/strided_slice_4/stack_2?
!generator_8/glu_8/strided_slice_4StridedSlice&generator_8/conv2d_49/BiasAdd:output:00generator_8/glu_8/strided_slice_4/stack:output:02generator_8/glu_8/strided_slice_4/stack_1:output:02generator_8/glu_8/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2#
!generator_8/glu_8/strided_slice_4?
'generator_8/glu_8/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2)
'generator_8/glu_8/strided_slice_5/stack?
)generator_8/glu_8/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2+
)generator_8/glu_8/strided_slice_5/stack_1?
)generator_8/glu_8/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_8/glu_8/strided_slice_5/stack_2?
!generator_8/glu_8/strided_slice_5StridedSlice&generator_8/conv2d_49/BiasAdd:output:00generator_8/glu_8/strided_slice_5/stack:output:02generator_8/glu_8/strided_slice_5/stack_1:output:02generator_8/glu_8/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2#
!generator_8/glu_8/strided_slice_5?
generator_8/glu_8/Sigmoid_2Sigmoid*generator_8/glu_8/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
generator_8/glu_8/Sigmoid_2?
generator_8/glu_8/mul_2Mul*generator_8/glu_8/strided_slice_4:output:0generator_8/glu_8/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
generator_8/glu_8/mul_2?
+generator_8/conv2d_50/Conv2D/ReadVariableOpReadVariableOp4generator_8_conv2d_50_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+generator_8/conv2d_50/Conv2D/ReadVariableOp?
generator_8/conv2d_50/Conv2DConv2Dgenerator_8/glu_8/mul_2:z:03generator_8/conv2d_50/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
generator_8/conv2d_50/Conv2D?
,generator_8/conv2d_50/BiasAdd/ReadVariableOpReadVariableOp5generator_8_conv2d_50_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,generator_8/conv2d_50/BiasAdd/ReadVariableOp?
generator_8/conv2d_50/BiasAddBiasAdd%generator_8/conv2d_50/Conv2D:output:04generator_8/conv2d_50/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
generator_8/conv2d_50/BiasAdd?
generator_8/conv2d_50/TanhTanh&generator_8/conv2d_50/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
generator_8/conv2d_50/Tanh?
IdentityIdentitygenerator_8/conv2d_50/Tanh:y:0-^generator_8/conv2d_48/BiasAdd/ReadVariableOp,^generator_8/conv2d_48/Conv2D/ReadVariableOp-^generator_8/conv2d_49/BiasAdd/ReadVariableOp,^generator_8/conv2d_49/Conv2D/ReadVariableOp-^generator_8/conv2d_50/BiasAdd/ReadVariableOp,^generator_8/conv2d_50/Conv2D/ReadVariableOp+^generator_8/dense_8/BiasAdd/ReadVariableOp*^generator_8/dense_8/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2\
,generator_8/conv2d_48/BiasAdd/ReadVariableOp,generator_8/conv2d_48/BiasAdd/ReadVariableOp2Z
+generator_8/conv2d_48/Conv2D/ReadVariableOp+generator_8/conv2d_48/Conv2D/ReadVariableOp2\
,generator_8/conv2d_49/BiasAdd/ReadVariableOp,generator_8/conv2d_49/BiasAdd/ReadVariableOp2Z
+generator_8/conv2d_49/Conv2D/ReadVariableOp+generator_8/conv2d_49/Conv2D/ReadVariableOp2\
,generator_8/conv2d_50/BiasAdd/ReadVariableOp,generator_8/conv2d_50/BiasAdd/ReadVariableOp2Z
+generator_8/conv2d_50/Conv2D/ReadVariableOp+generator_8/conv2d_50/Conv2D/ReadVariableOp2X
*generator_8/dense_8/BiasAdd/ReadVariableOp*generator_8/dense_8/BiasAdd/ReadVariableOp2V
)generator_8/dense_8/MatMul/ReadVariableOp)generator_8/dense_8/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
`
D__inference_glu_8_layer_call_and_return_conditional_losses_147418539

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
E
)__inference_glu_8_layer_call_fn_147418563

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
GPU2*0,1J 8? *M
fHRF
D__inference_glu_8_layer_call_and_return_conditional_losses_1474175922
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
_tf_keras_model?{"name": "generator_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Generator", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [128, 128]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Generator"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"?
_tf_keras_layer?{"name": "dense_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_8", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__"?
_tf_keras_model?{"name": "glu_8", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "GLU", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 12544]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "GLU"}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"name": "reshape_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_8", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 128]}}, "shared_object_id": 4}
?
trainable_variables
regularization_losses
	variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_layer?{"name": "up_sampling2d_8", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_8", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 6}}
?


kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?	
_tf_keras_layer?	{"name": "conv2d_48", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_48", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 128]}}
?


%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?	
_tf_keras_layer?	{"name": "conv2d_49", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_49", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 64]}}
?


+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
*h&call_and_return_all_conditional_losses
i__call__"?	
_tf_keras_layer?	{"name": "conv2d_50", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_50", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
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
??b2dense_8/kernel
:?b2dense_8/bias
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
,:*??2conv2d_48/kernel
:?2conv2d_48/bias
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
*:(@@2conv2d_49/kernel
:@2conv2d_49/bias
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
*:( 2conv2d_50/kernel
:2conv2d_50/bias
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
J__inference_generator_8_layer_call_and_return_conditional_losses_147418135
J__inference_generator_8_layer_call_and_return_conditional_losses_147418212
J__inference_generator_8_layer_call_and_return_conditional_losses_147418289
J__inference_generator_8_layer_call_and_return_conditional_losses_147418366?
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
/__inference_generator_8_layer_call_fn_147418387
/__inference_generator_8_layer_call_fn_147418408
/__inference_generator_8_layer_call_fn_147418429
/__inference_generator_8_layer_call_fn_147418450?
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
$__inference__wrapped_model_147417550?
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
F__inference_dense_8_layer_call_and_return_conditional_losses_147418460?
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
+__inference_dense_8_layer_call_fn_147418469?
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
D__inference_glu_8_layer_call_and_return_conditional_losses_147418483
D__inference_glu_8_layer_call_and_return_conditional_losses_147418497
D__inference_glu_8_layer_call_and_return_conditional_losses_147417614
D__inference_glu_8_layer_call_and_return_conditional_losses_147417628
D__inference_glu_8_layer_call_and_return_conditional_losses_147418511
D__inference_glu_8_layer_call_and_return_conditional_losses_147418525
D__inference_glu_8_layer_call_and_return_conditional_losses_147418539
D__inference_glu_8_layer_call_and_return_conditional_losses_147418553?
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
)__inference_glu_8_layer_call_fn_147417571
)__inference_glu_8_layer_call_fn_147418558
)__inference_glu_8_layer_call_fn_147418563
)__inference_glu_8_layer_call_fn_147417600
)__inference_glu_8_layer_call_fn_147418568
)__inference_glu_8_layer_call_fn_147418573
)__inference_glu_8_layer_call_fn_147418578
)__inference_glu_8_layer_call_fn_147418583?
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
H__inference_reshape_8_layer_call_and_return_conditional_losses_147418597?
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
-__inference_reshape_8_layer_call_fn_147418602?
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
N__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_147417641?
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
3__inference_up_sampling2d_8_layer_call_fn_147417647?
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
H__inference_conv2d_48_layer_call_and_return_conditional_losses_147418612?
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
-__inference_conv2d_48_layer_call_fn_147418621?
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
H__inference_conv2d_49_layer_call_and_return_conditional_losses_147418631?
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
-__inference_conv2d_49_layer_call_fn_147418640?
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
H__inference_conv2d_50_layer_call_and_return_conditional_losses_147418651?
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
-__inference_conv2d_50_layer_call_fn_147418660?
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
'__inference_signature_wrapper_147418058input_1"?
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
$__inference__wrapped_model_147417550z %&+,1?.
'?$
"?
input_1??????????
? ";?8
6
output_1*?'
output_1??????????
H__inference_conv2d_48_layer_call_and_return_conditional_losses_147418612? J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
-__inference_conv2d_48_layer_call_fn_147418621? J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
H__inference_conv2d_49_layer_call_and_return_conditional_losses_147418631?%&I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
-__inference_conv2d_49_layer_call_fn_147418640?%&I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
H__inference_conv2d_50_layer_call_and_return_conditional_losses_147418651?+,I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
-__inference_conv2d_50_layer_call_fn_147418660?+,I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
F__inference_dense_8_layer_call_and_return_conditional_losses_147418460^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????b
? ?
+__inference_dense_8_layer_call_fn_147418469Q0?-
&?#
!?
inputs??????????
? "???????????b?
J__inference_generator_8_layer_call_and_return_conditional_losses_147418135o %&+,4?1
*?'
!?
inputs??????????
p 
? "-?*
#? 
0?????????
? ?
J__inference_generator_8_layer_call_and_return_conditional_losses_147418212o %&+,4?1
*?'
!?
inputs??????????
p
? "-?*
#? 
0?????????
? ?
J__inference_generator_8_layer_call_and_return_conditional_losses_147418289p %&+,5?2
+?(
"?
input_1??????????
p 
? "-?*
#? 
0?????????
? ?
J__inference_generator_8_layer_call_and_return_conditional_losses_147418366p %&+,5?2
+?(
"?
input_1??????????
p
? "-?*
#? 
0?????????
? ?
/__inference_generator_8_layer_call_fn_147418387u %&+,5?2
+?(
"?
input_1??????????
p 
? "2?/+????????????????????????????
/__inference_generator_8_layer_call_fn_147418408t %&+,4?1
*?'
!?
inputs??????????
p 
? "2?/+????????????????????????????
/__inference_generator_8_layer_call_fn_147418429t %&+,4?1
*?'
!?
inputs??????????
p
? "2?/+????????????????????????????
/__inference_generator_8_layer_call_fn_147418450u %&+,5?2
+?(
"?
input_1??????????
p
? "2?/+????????????????????????????
D__inference_glu_8_layer_call_and_return_conditional_losses_147417614kA?>
'?$
"?
input_1??????????b
?

trainingp "&?#
?
0??????????1
? ?
D__inference_glu_8_layer_call_and_return_conditional_losses_147417628kA?>
'?$
"?
input_1??????????b
?

trainingp"&?#
?
0??????????1
? ?
D__inference_glu_8_layer_call_and_return_conditional_losses_147418483j@?=
&?#
!?
inputs??????????b
?

trainingp "&?#
?
0??????????1
? ?
D__inference_glu_8_layer_call_and_return_conditional_losses_147418497j@?=
&?#
!?
inputs??????????b
?

trainingp"&?#
?
0??????????1
? ?
D__inference_glu_8_layer_call_and_return_conditional_losses_147418511?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp "??<
5?2
0+???????????????????????????@
? ?
D__inference_glu_8_layer_call_and_return_conditional_losses_147418525?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp "??<
5?2
0+??????????????????????????? 
? ?
D__inference_glu_8_layer_call_and_return_conditional_losses_147418539?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp"??<
5?2
0+??????????????????????????? 
? ?
D__inference_glu_8_layer_call_and_return_conditional_losses_147418553?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp"??<
5?2
0+???????????????????????????@
? ?
)__inference_glu_8_layer_call_fn_147417571^A?>
'?$
"?
input_1??????????b
?

trainingp "???????????1?
)__inference_glu_8_layer_call_fn_147417600^A?>
'?$
"?
input_1??????????b
?

trainingp"???????????1?
)__inference_glu_8_layer_call_fn_147418558]@?=
&?#
!?
inputs??????????b
?

trainingp "???????????1?
)__inference_glu_8_layer_call_fn_147418563]@?=
&?#
!?
inputs??????????b
?

trainingp"???????????1?
)__inference_glu_8_layer_call_fn_147418568?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp "2?/+??????????????????????????? ?
)__inference_glu_8_layer_call_fn_147418573?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp"2?/+??????????????????????????? ?
)__inference_glu_8_layer_call_fn_147418578?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp "2?/+???????????????????????????@?
)__inference_glu_8_layer_call_fn_147418583?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp"2?/+???????????????????????????@?
H__inference_reshape_8_layer_call_and_return_conditional_losses_147418597b0?-
&?#
!?
inputs??????????1
? ".?+
$?!
0??????????
? ?
-__inference_reshape_8_layer_call_fn_147418602U0?-
&?#
!?
inputs??????????1
? "!????????????
'__inference_signature_wrapper_147418058? %&+,<?9
? 
2?/
-
input_1"?
input_1??????????";?8
6
output_1*?'
output_1??????????
N__inference_up_sampling2d_8_layer_call_and_return_conditional_losses_147417641?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_up_sampling2d_8_layer_call_fn_147417647?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????