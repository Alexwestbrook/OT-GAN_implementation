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
dense_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
??b*
shared_namedense_4/kernel
s
"dense_4/kernel/Read/ReadVariableOpReadVariableOpdense_4/kernel* 
_output_shapes
:
??b*
dtype0
q
dense_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?b*
shared_namedense_4/bias
j
 dense_4/bias/Read/ReadVariableOpReadVariableOpdense_4/bias*
_output_shapes	
:?b*
dtype0
?
conv2d_24/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_24/kernel

$conv2d_24/kernel/Read/ReadVariableOpReadVariableOpconv2d_24/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_24/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_24/bias
n
"conv2d_24/bias/Read/ReadVariableOpReadVariableOpconv2d_24/bias*
_output_shapes	
:?*
dtype0
?
conv2d_25/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_25/kernel
}
$conv2d_25/kernel/Read/ReadVariableOpReadVariableOpconv2d_25/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_25/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_25/bias
m
"conv2d_25/bias/Read/ReadVariableOpReadVariableOpconv2d_25/bias*
_output_shapes
:@*
dtype0
?
conv2d_26/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_26/kernel
}
$conv2d_26/kernel/Read/ReadVariableOpReadVariableOpconv2d_26/kernel*&
_output_shapes
: *
dtype0
t
conv2d_26/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d_26/bias
m
"conv2d_26/bias/Read/ReadVariableOpReadVariableOpconv2d_26/bias*
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
VARIABLE_VALUEdense_4/kernel'dense/kernel/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEdense_4/bias%dense/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_24/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_24/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_25/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_25/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_26/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_26/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense_4/kerneldense_4/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/bias*
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
'__inference_signature_wrapper_107884797
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename"dense_4/kernel/Read/ReadVariableOp dense_4/bias/Read/ReadVariableOp$conv2d_24/kernel/Read/ReadVariableOp"conv2d_24/bias/Read/ReadVariableOp$conv2d_25/kernel/Read/ReadVariableOp"conv2d_25/bias/Read/ReadVariableOp$conv2d_26/kernel/Read/ReadVariableOp"conv2d_26/bias/Read/ReadVariableOpConst*
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
"__inference__traced_save_107885446
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_4/kerneldense_4/biasconv2d_24/kernelconv2d_24/biasconv2d_25/kernelconv2d_25/biasconv2d_26/kernelconv2d_26/bias*
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
%__inference__traced_restore_107885480??	
?
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107884456

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
?
a
D__inference_glu_4_layer_call_and_return_conditional_losses_107884367
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
?
E
)__inference_glu_4_layer_call_fn_107885322

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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078845922
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
-__inference_conv2d_24_layer_call_fn_107885360

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
H__inference_conv2d_24_layer_call_and_return_conditional_losses_1078844372
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
d
H__inference_reshape_4_layer_call_and_return_conditional_losses_107884424

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
?
E
)__inference_glu_4_layer_call_fn_107885317

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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078844562
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
?
/__inference_generator_4_layer_call_fn_107885126
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
J__inference_generator_4_layer_call_and_return_conditional_losses_1078845082
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
?,
?
J__inference_generator_4_layer_call_and_return_conditional_losses_107884674

inputs%
dense_4_107884647:
??b 
dense_4_107884649:	?b/
conv2d_24_107884655:??"
conv2d_24_107884657:	?-
conv2d_25_107884662:@@!
conv2d_25_107884664:@-
conv2d_26_107884668: !
conv2d_26_107884670:
identity??!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_107884647dense_4_107884649*
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
F__inference_dense_4_layer_call_and_return_conditional_losses_1078844032!
dense_4/StatefulPartitionedCall?
glu_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078843312
glu_4/PartitionedCall?
reshape_4/PartitionedCallPartitionedCallglu_4/PartitionedCall:output:0*
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
H__inference_reshape_4_layer_call_and_return_conditional_losses_1078844242
reshape_4/PartitionedCall?
up_sampling2d_4/PartitionedCallPartitionedCall"reshape_4/PartitionedCall:output:0*
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
N__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1078843802!
up_sampling2d_4/PartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_24_107884655conv2d_24_107884657*
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
H__inference_conv2d_24_layer_call_and_return_conditional_losses_1078844372#
!conv2d_24/StatefulPartitionedCall?
glu_4/PartitionedCall_1PartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078845922
glu_4/PartitionedCall_1?
!up_sampling2d_4/PartitionedCall_1PartitionedCall glu_4/PartitionedCall_1:output:0*
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
N__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1078843802#
!up_sampling2d_4/PartitionedCall_1?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*up_sampling2d_4/PartitionedCall_1:output:0conv2d_25_107884662conv2d_25_107884664*
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
H__inference_conv2d_25_layer_call_and_return_conditional_losses_1078844692#
!conv2d_25/StatefulPartitionedCall?
glu_4/PartitionedCall_2PartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078845582
glu_4/PartitionedCall_2?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall glu_4/PartitionedCall_2:output:0conv2d_26_107884668conv2d_26_107884670*
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
H__inference_conv2d_26_layer_call_and_return_conditional_losses_1078845012#
!conv2d_26/StatefulPartitionedCall?
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
"__inference__traced_save_107885446
file_prefix-
)savev2_dense_4_kernel_read_readvariableop+
'savev2_dense_4_bias_read_readvariableop/
+savev2_conv2d_24_kernel_read_readvariableop-
)savev2_conv2d_24_bias_read_readvariableop/
+savev2_conv2d_25_kernel_read_readvariableop-
)savev2_conv2d_25_bias_read_readvariableop/
+savev2_conv2d_26_kernel_read_readvariableop-
)savev2_conv2d_26_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0)savev2_dense_4_kernel_read_readvariableop'savev2_dense_4_bias_read_readvariableop+savev2_conv2d_24_kernel_read_readvariableop)savev2_conv2d_24_bias_read_readvariableop+savev2_conv2d_25_kernel_read_readvariableop)savev2_conv2d_25_bias_read_readvariableop+savev2_conv2d_26_kernel_read_readvariableop)savev2_conv2d_26_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
$__inference__wrapped_model_107884289
input_1F
2generator_4_dense_4_matmul_readvariableop_resource:
??bB
3generator_4_dense_4_biasadd_readvariableop_resource:	?bP
4generator_4_conv2d_24_conv2d_readvariableop_resource:??D
5generator_4_conv2d_24_biasadd_readvariableop_resource:	?N
4generator_4_conv2d_25_conv2d_readvariableop_resource:@@C
5generator_4_conv2d_25_biasadd_readvariableop_resource:@N
4generator_4_conv2d_26_conv2d_readvariableop_resource: C
5generator_4_conv2d_26_biasadd_readvariableop_resource:
identity??,generator_4/conv2d_24/BiasAdd/ReadVariableOp?+generator_4/conv2d_24/Conv2D/ReadVariableOp?,generator_4/conv2d_25/BiasAdd/ReadVariableOp?+generator_4/conv2d_25/Conv2D/ReadVariableOp?,generator_4/conv2d_26/BiasAdd/ReadVariableOp?+generator_4/conv2d_26/Conv2D/ReadVariableOp?*generator_4/dense_4/BiasAdd/ReadVariableOp?)generator_4/dense_4/MatMul/ReadVariableOp?
)generator_4/dense_4/MatMul/ReadVariableOpReadVariableOp2generator_4_dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02+
)generator_4/dense_4/MatMul/ReadVariableOp?
generator_4/dense_4/MatMulMatMulinput_11generator_4/dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
generator_4/dense_4/MatMul?
*generator_4/dense_4/BiasAdd/ReadVariableOpReadVariableOp3generator_4_dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02,
*generator_4/dense_4/BiasAdd/ReadVariableOp?
generator_4/dense_4/BiasAddBiasAdd$generator_4/dense_4/MatMul:product:02generator_4/dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
generator_4/dense_4/BiasAdd?
%generator_4/glu_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2'
%generator_4/glu_4/strided_slice/stack?
'generator_4/glu_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'generator_4/glu_4/strided_slice/stack_1?
'generator_4/glu_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2)
'generator_4/glu_4/strided_slice/stack_2?
generator_4/glu_4/strided_sliceStridedSlice$generator_4/dense_4/BiasAdd:output:0.generator_4/glu_4/strided_slice/stack:output:00generator_4/glu_4/strided_slice/stack_1:output:00generator_4/glu_4/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2!
generator_4/glu_4/strided_slice?
'generator_4/glu_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2)
'generator_4/glu_4/strided_slice_1/stack?
)generator_4/glu_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2+
)generator_4/glu_4/strided_slice_1/stack_1?
)generator_4/glu_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2+
)generator_4/glu_4/strided_slice_1/stack_2?
!generator_4/glu_4/strided_slice_1StridedSlice$generator_4/dense_4/BiasAdd:output:00generator_4/glu_4/strided_slice_1/stack:output:02generator_4/glu_4/strided_slice_1/stack_1:output:02generator_4/glu_4/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2#
!generator_4/glu_4/strided_slice_1?
generator_4/glu_4/SigmoidSigmoid*generator_4/glu_4/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
generator_4/glu_4/Sigmoid?
generator_4/glu_4/mulMul(generator_4/glu_4/strided_slice:output:0generator_4/glu_4/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
generator_4/glu_4/mul?
generator_4/reshape_4/ShapeShapegenerator_4/glu_4/mul:z:0*
T0*
_output_shapes
:2
generator_4/reshape_4/Shape?
)generator_4/reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2+
)generator_4/reshape_4/strided_slice/stack?
+generator_4/reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2-
+generator_4/reshape_4/strided_slice/stack_1?
+generator_4/reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2-
+generator_4/reshape_4/strided_slice/stack_2?
#generator_4/reshape_4/strided_sliceStridedSlice$generator_4/reshape_4/Shape:output:02generator_4/reshape_4/strided_slice/stack:output:04generator_4/reshape_4/strided_slice/stack_1:output:04generator_4/reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2%
#generator_4/reshape_4/strided_slice?
%generator_4/reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2'
%generator_4/reshape_4/Reshape/shape/1?
%generator_4/reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2'
%generator_4/reshape_4/Reshape/shape/2?
%generator_4/reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2'
%generator_4/reshape_4/Reshape/shape/3?
#generator_4/reshape_4/Reshape/shapePack,generator_4/reshape_4/strided_slice:output:0.generator_4/reshape_4/Reshape/shape/1:output:0.generator_4/reshape_4/Reshape/shape/2:output:0.generator_4/reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2%
#generator_4/reshape_4/Reshape/shape?
generator_4/reshape_4/ReshapeReshapegenerator_4/glu_4/mul:z:0,generator_4/reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
generator_4/reshape_4/Reshape?
!generator_4/up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2#
!generator_4/up_sampling2d_4/Const?
#generator_4/up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator_4/up_sampling2d_4/Const_1?
generator_4/up_sampling2d_4/mulMul*generator_4/up_sampling2d_4/Const:output:0,generator_4/up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2!
generator_4/up_sampling2d_4/mul?
8generator_4/up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighbor&generator_4/reshape_4/Reshape:output:0#generator_4/up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2:
8generator_4/up_sampling2d_4/resize/ResizeNearestNeighbor?
+generator_4/conv2d_24/Conv2D/ReadVariableOpReadVariableOp4generator_4_conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02-
+generator_4/conv2d_24/Conv2D/ReadVariableOp?
generator_4/conv2d_24/Conv2DConv2DIgenerator_4/up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:03generator_4/conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
generator_4/conv2d_24/Conv2D?
,generator_4/conv2d_24/BiasAdd/ReadVariableOpReadVariableOp5generator_4_conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02.
,generator_4/conv2d_24/BiasAdd/ReadVariableOp?
generator_4/conv2d_24/BiasAddBiasAdd%generator_4/conv2d_24/Conv2D:output:04generator_4/conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
generator_4/conv2d_24/BiasAdd?
'generator_4/glu_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2)
'generator_4/glu_4/strided_slice_2/stack?
)generator_4/glu_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2+
)generator_4/glu_4/strided_slice_2/stack_1?
)generator_4/glu_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_4/glu_4/strided_slice_2/stack_2?
!generator_4/glu_4/strided_slice_2StridedSlice&generator_4/conv2d_24/BiasAdd:output:00generator_4/glu_4/strided_slice_2/stack:output:02generator_4/glu_4/strided_slice_2/stack_1:output:02generator_4/glu_4/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2#
!generator_4/glu_4/strided_slice_2?
'generator_4/glu_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2)
'generator_4/glu_4/strided_slice_3/stack?
)generator_4/glu_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2+
)generator_4/glu_4/strided_slice_3/stack_1?
)generator_4/glu_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_4/glu_4/strided_slice_3/stack_2?
!generator_4/glu_4/strided_slice_3StridedSlice&generator_4/conv2d_24/BiasAdd:output:00generator_4/glu_4/strided_slice_3/stack:output:02generator_4/glu_4/strided_slice_3/stack_1:output:02generator_4/glu_4/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2#
!generator_4/glu_4/strided_slice_3?
generator_4/glu_4/Sigmoid_1Sigmoid*generator_4/glu_4/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
generator_4/glu_4/Sigmoid_1?
generator_4/glu_4/mul_1Mul*generator_4/glu_4/strided_slice_2:output:0generator_4/glu_4/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
generator_4/glu_4/mul_1?
#generator_4/up_sampling2d_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator_4/up_sampling2d_4/Const_2?
#generator_4/up_sampling2d_4/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2%
#generator_4/up_sampling2d_4/Const_3?
!generator_4/up_sampling2d_4/mul_1Mul,generator_4/up_sampling2d_4/Const_2:output:0,generator_4/up_sampling2d_4/Const_3:output:0*
T0*
_output_shapes
:2#
!generator_4/up_sampling2d_4/mul_1?
:generator_4/up_sampling2d_4/resize_1/ResizeNearestNeighborResizeNearestNeighborgenerator_4/glu_4/mul_1:z:0%generator_4/up_sampling2d_4/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(2<
:generator_4/up_sampling2d_4/resize_1/ResizeNearestNeighbor?
+generator_4/conv2d_25/Conv2D/ReadVariableOpReadVariableOp4generator_4_conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02-
+generator_4/conv2d_25/Conv2D/ReadVariableOp?
generator_4/conv2d_25/Conv2DConv2DKgenerator_4/up_sampling2d_4/resize_1/ResizeNearestNeighbor:resized_images:03generator_4/conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
generator_4/conv2d_25/Conv2D?
,generator_4/conv2d_25/BiasAdd/ReadVariableOpReadVariableOp5generator_4_conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02.
,generator_4/conv2d_25/BiasAdd/ReadVariableOp?
generator_4/conv2d_25/BiasAddBiasAdd%generator_4/conv2d_25/Conv2D:output:04generator_4/conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
generator_4/conv2d_25/BiasAdd?
'generator_4/glu_4/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2)
'generator_4/glu_4/strided_slice_4/stack?
)generator_4/glu_4/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2+
)generator_4/glu_4/strided_slice_4/stack_1?
)generator_4/glu_4/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_4/glu_4/strided_slice_4/stack_2?
!generator_4/glu_4/strided_slice_4StridedSlice&generator_4/conv2d_25/BiasAdd:output:00generator_4/glu_4/strided_slice_4/stack:output:02generator_4/glu_4/strided_slice_4/stack_1:output:02generator_4/glu_4/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2#
!generator_4/glu_4/strided_slice_4?
'generator_4/glu_4/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2)
'generator_4/glu_4/strided_slice_5/stack?
)generator_4/glu_4/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2+
)generator_4/glu_4/strided_slice_5/stack_1?
)generator_4/glu_4/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2+
)generator_4/glu_4/strided_slice_5/stack_2?
!generator_4/glu_4/strided_slice_5StridedSlice&generator_4/conv2d_25/BiasAdd:output:00generator_4/glu_4/strided_slice_5/stack:output:02generator_4/glu_4/strided_slice_5/stack_1:output:02generator_4/glu_4/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2#
!generator_4/glu_4/strided_slice_5?
generator_4/glu_4/Sigmoid_2Sigmoid*generator_4/glu_4/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
generator_4/glu_4/Sigmoid_2?
generator_4/glu_4/mul_2Mul*generator_4/glu_4/strided_slice_4:output:0generator_4/glu_4/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
generator_4/glu_4/mul_2?
+generator_4/conv2d_26/Conv2D/ReadVariableOpReadVariableOp4generator_4_conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02-
+generator_4/conv2d_26/Conv2D/ReadVariableOp?
generator_4/conv2d_26/Conv2DConv2Dgenerator_4/glu_4/mul_2:z:03generator_4/conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
generator_4/conv2d_26/Conv2D?
,generator_4/conv2d_26/BiasAdd/ReadVariableOpReadVariableOp5generator_4_conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02.
,generator_4/conv2d_26/BiasAdd/ReadVariableOp?
generator_4/conv2d_26/BiasAddBiasAdd%generator_4/conv2d_26/Conv2D:output:04generator_4/conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
generator_4/conv2d_26/BiasAdd?
generator_4/conv2d_26/TanhTanh&generator_4/conv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
generator_4/conv2d_26/Tanh?
IdentityIdentitygenerator_4/conv2d_26/Tanh:y:0-^generator_4/conv2d_24/BiasAdd/ReadVariableOp,^generator_4/conv2d_24/Conv2D/ReadVariableOp-^generator_4/conv2d_25/BiasAdd/ReadVariableOp,^generator_4/conv2d_25/Conv2D/ReadVariableOp-^generator_4/conv2d_26/BiasAdd/ReadVariableOp,^generator_4/conv2d_26/Conv2D/ReadVariableOp+^generator_4/dense_4/BiasAdd/ReadVariableOp*^generator_4/dense_4/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2\
,generator_4/conv2d_24/BiasAdd/ReadVariableOp,generator_4/conv2d_24/BiasAdd/ReadVariableOp2Z
+generator_4/conv2d_24/Conv2D/ReadVariableOp+generator_4/conv2d_24/Conv2D/ReadVariableOp2\
,generator_4/conv2d_25/BiasAdd/ReadVariableOp,generator_4/conv2d_25/BiasAdd/ReadVariableOp2Z
+generator_4/conv2d_25/Conv2D/ReadVariableOp+generator_4/conv2d_25/Conv2D/ReadVariableOp2\
,generator_4/conv2d_26/BiasAdd/ReadVariableOp,generator_4/conv2d_26/BiasAdd/ReadVariableOp2Z
+generator_4/conv2d_26/Conv2D/ReadVariableOp+generator_4/conv2d_26/Conv2D/ReadVariableOp2X
*generator_4/dense_4/BiasAdd/ReadVariableOp*generator_4/dense_4/BiasAdd/ReadVariableOp2V
)generator_4/dense_4/MatMul/ReadVariableOp)generator_4/dense_4/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?	
?
/__inference_generator_4_layer_call_fn_107885189
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
J__inference_generator_4_layer_call_and_return_conditional_losses_1078846742
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
a
D__inference_glu_4_layer_call_and_return_conditional_losses_107884353
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
D__inference_glu_4_layer_call_and_return_conditional_losses_107885250

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
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107885264

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
H__inference_conv2d_26_layer_call_and_return_conditional_losses_107884501

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
E
)__inference_glu_4_layer_call_fn_107885307

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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078844882
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
E
)__inference_glu_4_layer_call_fn_107885297

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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078843072
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
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107885236

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
?
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107884307

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
H__inference_conv2d_24_layer_call_and_return_conditional_losses_107885351

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
F
)__inference_glu_4_layer_call_fn_107884310
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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078843072
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
?h
?
J__inference_generator_4_layer_call_and_return_conditional_losses_107885105
input_1:
&dense_4_matmul_readvariableop_resource:
??b6
'dense_4_biasadd_readvariableop_resource:	?bD
(conv2d_24_conv2d_readvariableop_resource:??8
)conv2d_24_biasadd_readvariableop_resource:	?B
(conv2d_25_conv2d_readvariableop_resource:@@7
)conv2d_25_biasadd_readvariableop_resource:@B
(conv2d_26_conv2d_readvariableop_resource: 7
)conv2d_26_biasadd_readvariableop_resource:
identity?? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinput_1%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_4/BiasAdd?
glu_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_4/strided_slice/stack?
glu_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_4/strided_slice/stack_1?
glu_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_4/strided_slice/stack_2?
glu_4/strided_sliceStridedSlicedense_4/BiasAdd:output:0"glu_4/strided_slice/stack:output:0$glu_4/strided_slice/stack_1:output:0$glu_4/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_4/strided_slice?
glu_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_4/strided_slice_1/stack?
glu_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_4/strided_slice_1/stack_1?
glu_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_4/strided_slice_1/stack_2?
glu_4/strided_slice_1StridedSlicedense_4/BiasAdd:output:0$glu_4/strided_slice_1/stack:output:0&glu_4/strided_slice_1/stack_1:output:0&glu_4/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_4/strided_slice_1|
glu_4/SigmoidSigmoidglu_4/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_4/Sigmoid?
	glu_4/mulMulglu_4/strided_slice:output:0glu_4/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_4/mul_
reshape_4/ShapeShapeglu_4/mul:z:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2y
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_4/Reshape/shape/3?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshapeglu_4/mul:z:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_4/Reshape
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_1?
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborreshape_4/Reshape:output:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_24/BiasAdd?
glu_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_2/stack?
glu_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_4/strided_slice_2/stack_1?
glu_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_2/stack_2?
glu_4/strided_slice_2StridedSliceconv2d_24/BiasAdd:output:0$glu_4/strided_slice_2/stack:output:0&glu_4/strided_slice_2/stack_1:output:0&glu_4/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_4/strided_slice_2?
glu_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_4/strided_slice_3/stack?
glu_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_3/stack_1?
glu_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_3/stack_2?
glu_4/strided_slice_3StridedSliceconv2d_24/BiasAdd:output:0$glu_4/strided_slice_3/stack:output:0&glu_4/strided_slice_3/stack_1:output:0&glu_4/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_4/strided_slice_3?
glu_4/Sigmoid_1Sigmoidglu_4/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_4/Sigmoid_1?
glu_4/mul_1Mulglu_4/strided_slice_2:output:0glu_4/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_4/mul_1?
up_sampling2d_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_2?
up_sampling2d_4/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_3?
up_sampling2d_4/mul_1Mul up_sampling2d_4/Const_2:output:0 up_sampling2d_4/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul_1?
.up_sampling2d_4/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_4/mul_1:z:0up_sampling2d_4/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_4/resize_1/ResizeNearestNeighbor?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2D?up_sampling2d_4/resize_1/ResizeNearestNeighbor:resized_images:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_25/BiasAdd?
glu_4/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_4/stack?
glu_4/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_4/stack_1?
glu_4/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_4/stack_2?
glu_4/strided_slice_4StridedSliceconv2d_25/BiasAdd:output:0$glu_4/strided_slice_4/stack:output:0&glu_4/strided_slice_4/stack_1:output:0&glu_4/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_4/strided_slice_4?
glu_4/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_5/stack?
glu_4/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_5/stack_1?
glu_4/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_5/stack_2?
glu_4/strided_slice_5StridedSliceconv2d_25/BiasAdd:output:0$glu_4/strided_slice_5/stack:output:0&glu_4/strided_slice_5/stack_1:output:0&glu_4/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_4/strided_slice_5?
glu_4/Sigmoid_2Sigmoidglu_4/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_4/Sigmoid_2?
glu_4/mul_2Mulglu_4/strided_slice_4:output:0glu_4/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_4/mul_2?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dglu_4/mul_2:z:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_26/BiasAdd~
conv2d_26/TanhTanhconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_26/Tanh?
IdentityIdentityconv2d_26/Tanh:y:0!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?h
?
J__inference_generator_4_layer_call_and_return_conditional_losses_107884874

inputs:
&dense_4_matmul_readvariableop_resource:
??b6
'dense_4_biasadd_readvariableop_resource:	?bD
(conv2d_24_conv2d_readvariableop_resource:??8
)conv2d_24_biasadd_readvariableop_resource:	?B
(conv2d_25_conv2d_readvariableop_resource:@@7
)conv2d_25_biasadd_readvariableop_resource:@B
(conv2d_26_conv2d_readvariableop_resource: 7
)conv2d_26_biasadd_readvariableop_resource:
identity?? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_4/BiasAdd?
glu_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_4/strided_slice/stack?
glu_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_4/strided_slice/stack_1?
glu_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_4/strided_slice/stack_2?
glu_4/strided_sliceStridedSlicedense_4/BiasAdd:output:0"glu_4/strided_slice/stack:output:0$glu_4/strided_slice/stack_1:output:0$glu_4/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_4/strided_slice?
glu_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_4/strided_slice_1/stack?
glu_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_4/strided_slice_1/stack_1?
glu_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_4/strided_slice_1/stack_2?
glu_4/strided_slice_1StridedSlicedense_4/BiasAdd:output:0$glu_4/strided_slice_1/stack:output:0&glu_4/strided_slice_1/stack_1:output:0&glu_4/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_4/strided_slice_1|
glu_4/SigmoidSigmoidglu_4/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_4/Sigmoid?
	glu_4/mulMulglu_4/strided_slice:output:0glu_4/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_4/mul_
reshape_4/ShapeShapeglu_4/mul:z:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2y
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_4/Reshape/shape/3?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshapeglu_4/mul:z:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_4/Reshape
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_1?
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborreshape_4/Reshape:output:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_24/BiasAdd?
glu_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_2/stack?
glu_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_4/strided_slice_2/stack_1?
glu_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_2/stack_2?
glu_4/strided_slice_2StridedSliceconv2d_24/BiasAdd:output:0$glu_4/strided_slice_2/stack:output:0&glu_4/strided_slice_2/stack_1:output:0&glu_4/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_4/strided_slice_2?
glu_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_4/strided_slice_3/stack?
glu_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_3/stack_1?
glu_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_3/stack_2?
glu_4/strided_slice_3StridedSliceconv2d_24/BiasAdd:output:0$glu_4/strided_slice_3/stack:output:0&glu_4/strided_slice_3/stack_1:output:0&glu_4/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_4/strided_slice_3?
glu_4/Sigmoid_1Sigmoidglu_4/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_4/Sigmoid_1?
glu_4/mul_1Mulglu_4/strided_slice_2:output:0glu_4/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_4/mul_1?
up_sampling2d_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_2?
up_sampling2d_4/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_3?
up_sampling2d_4/mul_1Mul up_sampling2d_4/Const_2:output:0 up_sampling2d_4/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul_1?
.up_sampling2d_4/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_4/mul_1:z:0up_sampling2d_4/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_4/resize_1/ResizeNearestNeighbor?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2D?up_sampling2d_4/resize_1/ResizeNearestNeighbor:resized_images:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_25/BiasAdd?
glu_4/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_4/stack?
glu_4/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_4/stack_1?
glu_4/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_4/stack_2?
glu_4/strided_slice_4StridedSliceconv2d_25/BiasAdd:output:0$glu_4/strided_slice_4/stack:output:0&glu_4/strided_slice_4/stack_1:output:0&glu_4/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_4/strided_slice_4?
glu_4/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_5/stack?
glu_4/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_5/stack_1?
glu_4/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_5/stack_2?
glu_4/strided_slice_5StridedSliceconv2d_25/BiasAdd:output:0$glu_4/strided_slice_5/stack:output:0&glu_4/strided_slice_5/stack_1:output:0&glu_4/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_4/strided_slice_5?
glu_4/Sigmoid_2Sigmoidglu_4/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_4/Sigmoid_2?
glu_4/mul_2Mulglu_4/strided_slice_4:output:0glu_4/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_4/mul_2?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dglu_4/mul_2:z:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_26/BiasAdd~
conv2d_26/TanhTanhconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_26/Tanh?
IdentityIdentityconv2d_26/Tanh:y:0!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
E
)__inference_glu_4_layer_call_fn_107885312

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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078845582
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
I
-__inference_reshape_4_layer_call_fn_107885341

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
H__inference_reshape_4_layer_call_and_return_conditional_losses_1078844242
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
?
O
3__inference_up_sampling2d_4_layer_call_fn_107884386

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
N__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1078843802
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
?
F__inference_dense_4_layer_call_and_return_conditional_losses_107885199

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
-__inference_conv2d_25_layer_call_fn_107885379

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
H__inference_conv2d_25_layer_call_and_return_conditional_losses_1078844692
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
?
F
)__inference_glu_4_layer_call_fn_107884339
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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078843312
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
H__inference_conv2d_26_layer_call_and_return_conditional_losses_107885390

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
H__inference_conv2d_25_layer_call_and_return_conditional_losses_107884469

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
?
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107885222

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
j
N__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_107884380

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
?h
?
J__inference_generator_4_layer_call_and_return_conditional_losses_107885028
input_1:
&dense_4_matmul_readvariableop_resource:
??b6
'dense_4_biasadd_readvariableop_resource:	?bD
(conv2d_24_conv2d_readvariableop_resource:??8
)conv2d_24_biasadd_readvariableop_resource:	?B
(conv2d_25_conv2d_readvariableop_resource:@@7
)conv2d_25_biasadd_readvariableop_resource:@B
(conv2d_26_conv2d_readvariableop_resource: 7
)conv2d_26_biasadd_readvariableop_resource:
identity?? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinput_1%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_4/BiasAdd?
glu_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_4/strided_slice/stack?
glu_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_4/strided_slice/stack_1?
glu_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_4/strided_slice/stack_2?
glu_4/strided_sliceStridedSlicedense_4/BiasAdd:output:0"glu_4/strided_slice/stack:output:0$glu_4/strided_slice/stack_1:output:0$glu_4/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_4/strided_slice?
glu_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_4/strided_slice_1/stack?
glu_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_4/strided_slice_1/stack_1?
glu_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_4/strided_slice_1/stack_2?
glu_4/strided_slice_1StridedSlicedense_4/BiasAdd:output:0$glu_4/strided_slice_1/stack:output:0&glu_4/strided_slice_1/stack_1:output:0&glu_4/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_4/strided_slice_1|
glu_4/SigmoidSigmoidglu_4/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_4/Sigmoid?
	glu_4/mulMulglu_4/strided_slice:output:0glu_4/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_4/mul_
reshape_4/ShapeShapeglu_4/mul:z:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2y
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_4/Reshape/shape/3?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshapeglu_4/mul:z:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_4/Reshape
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_1?
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborreshape_4/Reshape:output:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_24/BiasAdd?
glu_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_2/stack?
glu_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_4/strided_slice_2/stack_1?
glu_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_2/stack_2?
glu_4/strided_slice_2StridedSliceconv2d_24/BiasAdd:output:0$glu_4/strided_slice_2/stack:output:0&glu_4/strided_slice_2/stack_1:output:0&glu_4/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_4/strided_slice_2?
glu_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_4/strided_slice_3/stack?
glu_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_3/stack_1?
glu_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_3/stack_2?
glu_4/strided_slice_3StridedSliceconv2d_24/BiasAdd:output:0$glu_4/strided_slice_3/stack:output:0&glu_4/strided_slice_3/stack_1:output:0&glu_4/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_4/strided_slice_3?
glu_4/Sigmoid_1Sigmoidglu_4/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_4/Sigmoid_1?
glu_4/mul_1Mulglu_4/strided_slice_2:output:0glu_4/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_4/mul_1?
up_sampling2d_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_2?
up_sampling2d_4/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_3?
up_sampling2d_4/mul_1Mul up_sampling2d_4/Const_2:output:0 up_sampling2d_4/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul_1?
.up_sampling2d_4/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_4/mul_1:z:0up_sampling2d_4/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_4/resize_1/ResizeNearestNeighbor?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2D?up_sampling2d_4/resize_1/ResizeNearestNeighbor:resized_images:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_25/BiasAdd?
glu_4/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_4/stack?
glu_4/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_4/stack_1?
glu_4/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_4/stack_2?
glu_4/strided_slice_4StridedSliceconv2d_25/BiasAdd:output:0$glu_4/strided_slice_4/stack:output:0&glu_4/strided_slice_4/stack_1:output:0&glu_4/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_4/strided_slice_4?
glu_4/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_5/stack?
glu_4/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_5/stack_1?
glu_4/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_5/stack_2?
glu_4/strided_slice_5StridedSliceconv2d_25/BiasAdd:output:0$glu_4/strided_slice_5/stack:output:0&glu_4/strided_slice_5/stack_1:output:0&glu_4/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_4/strided_slice_5?
glu_4/Sigmoid_2Sigmoidglu_4/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_4/Sigmoid_2?
glu_4/mul_2Mulglu_4/strided_slice_4:output:0glu_4/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_4/mul_2?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dglu_4/mul_2:z:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_26/BiasAdd~
conv2d_26/TanhTanhconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_26/Tanh?
IdentityIdentityconv2d_26/Tanh:y:0!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:Q M
(
_output_shapes
:??????????
!
_user_specified_name	input_1
?
?
+__inference_dense_4_layer_call_fn_107885208

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
F__inference_dense_4_layer_call_and_return_conditional_losses_1078844032
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
?
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107884558

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
H__inference_conv2d_25_layer_call_and_return_conditional_losses_107885370

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
?
?
-__inference_conv2d_26_layer_call_fn_107885399

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
H__inference_conv2d_26_layer_call_and_return_conditional_losses_1078845012
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
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107885278

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
D__inference_glu_4_layer_call_and_return_conditional_losses_107884488

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
?,
?
J__inference_generator_4_layer_call_and_return_conditional_losses_107884508

inputs%
dense_4_107884404:
??b 
dense_4_107884406:	?b/
conv2d_24_107884438:??"
conv2d_24_107884440:	?-
conv2d_25_107884470:@@!
conv2d_25_107884472:@-
conv2d_26_107884502: !
conv2d_26_107884504:
identity??!conv2d_24/StatefulPartitionedCall?!conv2d_25/StatefulPartitionedCall?!conv2d_26/StatefulPartitionedCall?dense_4/StatefulPartitionedCall?
dense_4/StatefulPartitionedCallStatefulPartitionedCallinputsdense_4_107884404dense_4_107884406*
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
F__inference_dense_4_layer_call_and_return_conditional_losses_1078844032!
dense_4/StatefulPartitionedCall?
glu_4/PartitionedCallPartitionedCall(dense_4/StatefulPartitionedCall:output:0*
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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078843072
glu_4/PartitionedCall?
reshape_4/PartitionedCallPartitionedCallglu_4/PartitionedCall:output:0*
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
H__inference_reshape_4_layer_call_and_return_conditional_losses_1078844242
reshape_4/PartitionedCall?
up_sampling2d_4/PartitionedCallPartitionedCall"reshape_4/PartitionedCall:output:0*
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
N__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1078843802!
up_sampling2d_4/PartitionedCall?
!conv2d_24/StatefulPartitionedCallStatefulPartitionedCall(up_sampling2d_4/PartitionedCall:output:0conv2d_24_107884438conv2d_24_107884440*
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
H__inference_conv2d_24_layer_call_and_return_conditional_losses_1078844372#
!conv2d_24/StatefulPartitionedCall?
glu_4/PartitionedCall_1PartitionedCall*conv2d_24/StatefulPartitionedCall:output:0*
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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078844562
glu_4/PartitionedCall_1?
!up_sampling2d_4/PartitionedCall_1PartitionedCall glu_4/PartitionedCall_1:output:0*
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
N__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_1078843802#
!up_sampling2d_4/PartitionedCall_1?
!conv2d_25/StatefulPartitionedCallStatefulPartitionedCall*up_sampling2d_4/PartitionedCall_1:output:0conv2d_25_107884470conv2d_25_107884472*
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
H__inference_conv2d_25_layer_call_and_return_conditional_losses_1078844692#
!conv2d_25/StatefulPartitionedCall?
glu_4/PartitionedCall_2PartitionedCall*conv2d_25/StatefulPartitionedCall:output:0*
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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078844882
glu_4/PartitionedCall_2?
!conv2d_26/StatefulPartitionedCallStatefulPartitionedCall glu_4/PartitionedCall_2:output:0conv2d_26_107884502conv2d_26_107884504*
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
H__inference_conv2d_26_layer_call_and_return_conditional_losses_1078845012#
!conv2d_26/StatefulPartitionedCall?
IdentityIdentity*conv2d_26/StatefulPartitionedCall:output:0"^conv2d_24/StatefulPartitionedCall"^conv2d_25/StatefulPartitionedCall"^conv2d_26/StatefulPartitionedCall ^dense_4/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2F
!conv2d_24/StatefulPartitionedCall!conv2d_24/StatefulPartitionedCall2F
!conv2d_25/StatefulPartitionedCall!conv2d_25/StatefulPartitionedCall2F
!conv2d_26/StatefulPartitionedCall!conv2d_26/StatefulPartitionedCall2B
dense_4/StatefulPartitionedCalldense_4/StatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?%
?
%__inference__traced_restore_107885480
file_prefix3
assignvariableop_dense_4_kernel:
??b.
assignvariableop_1_dense_4_bias:	?b?
#assignvariableop_2_conv2d_24_kernel:??0
!assignvariableop_3_conv2d_24_bias:	?=
#assignvariableop_4_conv2d_25_kernel:@@/
!assignvariableop_5_conv2d_25_bias:@=
#assignvariableop_6_conv2d_26_kernel: /
!assignvariableop_7_conv2d_26_bias:

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
AssignVariableOpAssignVariableOpassignvariableop_dense_4_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_4_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_24_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_24_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_25_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_25_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp#assignvariableop_6_conv2d_26_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp!assignvariableop_7_conv2d_26_biasIdentity_7:output:0"/device:CPU:0*
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
?	
?
/__inference_generator_4_layer_call_fn_107885147

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
J__inference_generator_4_layer_call_and_return_conditional_losses_1078845082
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
?
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107884331

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
'__inference_signature_wrapper_107884797
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
$__inference__wrapped_model_1078842892
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
?h
?
J__inference_generator_4_layer_call_and_return_conditional_losses_107884951

inputs:
&dense_4_matmul_readvariableop_resource:
??b6
'dense_4_biasadd_readvariableop_resource:	?bD
(conv2d_24_conv2d_readvariableop_resource:??8
)conv2d_24_biasadd_readvariableop_resource:	?B
(conv2d_25_conv2d_readvariableop_resource:@@7
)conv2d_25_biasadd_readvariableop_resource:@B
(conv2d_26_conv2d_readvariableop_resource: 7
)conv2d_26_biasadd_readvariableop_resource:
identity?? conv2d_24/BiasAdd/ReadVariableOp?conv2d_24/Conv2D/ReadVariableOp? conv2d_25/BiasAdd/ReadVariableOp?conv2d_25/Conv2D/ReadVariableOp? conv2d_26/BiasAdd/ReadVariableOp?conv2d_26/Conv2D/ReadVariableOp?dense_4/BiasAdd/ReadVariableOp?dense_4/MatMul/ReadVariableOp?
dense_4/MatMul/ReadVariableOpReadVariableOp&dense_4_matmul_readvariableop_resource* 
_output_shapes
:
??b*
dtype02
dense_4/MatMul/ReadVariableOp?
dense_4/MatMulMatMulinputs%dense_4/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_4/MatMul?
dense_4/BiasAdd/ReadVariableOpReadVariableOp'dense_4_biasadd_readvariableop_resource*
_output_shapes	
:?b*
dtype02 
dense_4/BiasAdd/ReadVariableOp?
dense_4/BiasAddBiasAdddense_4/MatMul:product:0&dense_4/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????b2
dense_4/BiasAdd?
glu_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB"        2
glu_4/strided_slice/stack?
glu_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_4/strided_slice/stack_1?
glu_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_4/strided_slice/stack_2?
glu_4/strided_sliceStridedSlicedense_4/BiasAdd:output:0"glu_4/strided_slice/stack:output:0$glu_4/strided_slice/stack_1:output:0$glu_4/strided_slice/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_4/strided_slice?
glu_4/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB"    ?  2
glu_4/strided_slice_1/stack?
glu_4/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB"        2
glu_4/strided_slice_1/stack_1?
glu_4/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB"      2
glu_4/strided_slice_1/stack_2?
glu_4/strided_slice_1StridedSlicedense_4/BiasAdd:output:0$glu_4/strided_slice_1/stack:output:0&glu_4/strided_slice_1/stack_1:output:0&glu_4/strided_slice_1/stack_2:output:0*
Index0*
T0*(
_output_shapes
:??????????1*

begin_mask*
end_mask2
glu_4/strided_slice_1|
glu_4/SigmoidSigmoidglu_4/strided_slice_1:output:0*
T0*(
_output_shapes
:??????????12
glu_4/Sigmoid?
	glu_4/mulMulglu_4/strided_slice:output:0glu_4/Sigmoid:y:0*
T0*(
_output_shapes
:??????????12
	glu_4/mul_
reshape_4/ShapeShapeglu_4/mul:z:0*
T0*
_output_shapes
:2
reshape_4/Shape?
reshape_4/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape_4/strided_slice/stack?
reshape_4/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_1?
reshape_4/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2!
reshape_4/strided_slice/stack_2?
reshape_4/strided_sliceStridedSlicereshape_4/Shape:output:0&reshape_4/strided_slice/stack:output:0(reshape_4/strided_slice/stack_1:output:0(reshape_4/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape_4/strided_slicex
reshape_4/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/1x
reshape_4/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape_4/Reshape/shape/2y
reshape_4/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value
B :?2
reshape_4/Reshape/shape/3?
reshape_4/Reshape/shapePack reshape_4/strided_slice:output:0"reshape_4/Reshape/shape/1:output:0"reshape_4/Reshape/shape/2:output:0"reshape_4/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape_4/Reshape/shape?
reshape_4/ReshapeReshapeglu_4/mul:z:0 reshape_4/Reshape/shape:output:0*
T0*0
_output_shapes
:??????????2
reshape_4/Reshape
up_sampling2d_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const?
up_sampling2d_4/Const_1Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_1?
up_sampling2d_4/mulMulup_sampling2d_4/Const:output:0 up_sampling2d_4/Const_1:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul?
,up_sampling2d_4/resize/ResizeNearestNeighborResizeNearestNeighborreshape_4/Reshape:output:0up_sampling2d_4/mul:z:0*
T0*0
_output_shapes
:??????????*
half_pixel_centers(2.
,up_sampling2d_4/resize/ResizeNearestNeighbor?
conv2d_24/Conv2D/ReadVariableOpReadVariableOp(conv2d_24_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_24/Conv2D/ReadVariableOp?
conv2d_24/Conv2DConv2D=up_sampling2d_4/resize/ResizeNearestNeighbor:resized_images:0'conv2d_24/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_24/Conv2D?
 conv2d_24/BiasAdd/ReadVariableOpReadVariableOp)conv2d_24_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_24/BiasAdd/ReadVariableOp?
conv2d_24/BiasAddBiasAddconv2d_24/Conv2D:output:0(conv2d_24/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_24/BiasAdd?
glu_4/strided_slice_2/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_2/stack?
glu_4/strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_4/strided_slice_2/stack_1?
glu_4/strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_2/stack_2?
glu_4/strided_slice_2StridedSliceconv2d_24/BiasAdd:output:0$glu_4/strided_slice_2/stack:output:0&glu_4/strided_slice_2/stack_1:output:0&glu_4/strided_slice_2/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_4/strided_slice_2?
glu_4/strided_slice_3/stackConst*
_output_shapes
:*
dtype0*%
valueB"            @   2
glu_4/strided_slice_3/stack?
glu_4/strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_3/stack_1?
glu_4/strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_3/stack_2?
glu_4/strided_slice_3StridedSliceconv2d_24/BiasAdd:output:0$glu_4/strided_slice_3/stack:output:0&glu_4/strided_slice_3/stack_1:output:0&glu_4/strided_slice_3/stack_2:output:0*
Index0*
T0*/
_output_shapes
:?????????@*

begin_mask*
end_mask2
glu_4/strided_slice_3?
glu_4/Sigmoid_1Sigmoidglu_4/strided_slice_3:output:0*
T0*/
_output_shapes
:?????????@2
glu_4/Sigmoid_1?
glu_4/mul_1Mulglu_4/strided_slice_2:output:0glu_4/Sigmoid_1:y:0*
T0*/
_output_shapes
:?????????@2
glu_4/mul_1?
up_sampling2d_4/Const_2Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_2?
up_sampling2d_4/Const_3Const*
_output_shapes
:*
dtype0*
valueB"      2
up_sampling2d_4/Const_3?
up_sampling2d_4/mul_1Mul up_sampling2d_4/Const_2:output:0 up_sampling2d_4/Const_3:output:0*
T0*
_output_shapes
:2
up_sampling2d_4/mul_1?
.up_sampling2d_4/resize_1/ResizeNearestNeighborResizeNearestNeighborglu_4/mul_1:z:0up_sampling2d_4/mul_1:z:0*
T0*/
_output_shapes
:?????????@*
half_pixel_centers(20
.up_sampling2d_4/resize_1/ResizeNearestNeighbor?
conv2d_25/Conv2D/ReadVariableOpReadVariableOp(conv2d_25_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_25/Conv2D/ReadVariableOp?
conv2d_25/Conv2DConv2D?up_sampling2d_4/resize_1/ResizeNearestNeighbor:resized_images:0'conv2d_25/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_25/Conv2D?
 conv2d_25/BiasAdd/ReadVariableOpReadVariableOp)conv2d_25_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_25/BiasAdd/ReadVariableOp?
conv2d_25/BiasAddBiasAddconv2d_25/Conv2D:output:0(conv2d_25/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_25/BiasAdd?
glu_4/strided_slice_4/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_4/stack?
glu_4/strided_slice_4/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_4/stack_1?
glu_4/strided_slice_4/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_4/stack_2?
glu_4/strided_slice_4StridedSliceconv2d_25/BiasAdd:output:0$glu_4/strided_slice_4/stack:output:0&glu_4/strided_slice_4/stack_1:output:0&glu_4/strided_slice_4/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_4/strided_slice_4?
glu_4/strided_slice_5/stackConst*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_5/stack?
glu_4/strided_slice_5/stack_1Const*
_output_shapes
:*
dtype0*%
valueB"                2
glu_4/strided_slice_5/stack_1?
glu_4/strided_slice_5/stack_2Const*
_output_shapes
:*
dtype0*%
valueB"            2
glu_4/strided_slice_5/stack_2?
glu_4/strided_slice_5StridedSliceconv2d_25/BiasAdd:output:0$glu_4/strided_slice_5/stack:output:0&glu_4/strided_slice_5/stack_1:output:0&glu_4/strided_slice_5/stack_2:output:0*
Index0*
T0*/
_output_shapes
:????????? *

begin_mask*
end_mask2
glu_4/strided_slice_5?
glu_4/Sigmoid_2Sigmoidglu_4/strided_slice_5:output:0*
T0*/
_output_shapes
:????????? 2
glu_4/Sigmoid_2?
glu_4/mul_2Mulglu_4/strided_slice_4:output:0glu_4/Sigmoid_2:y:0*
T0*/
_output_shapes
:????????? 2
glu_4/mul_2?
conv2d_26/Conv2D/ReadVariableOpReadVariableOp(conv2d_26_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_26/Conv2D/ReadVariableOp?
conv2d_26/Conv2DConv2Dglu_4/mul_2:z:0'conv2d_26/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2
conv2d_26/Conv2D?
 conv2d_26/BiasAdd/ReadVariableOpReadVariableOp)conv2d_26_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02"
 conv2d_26/BiasAdd/ReadVariableOp?
conv2d_26/BiasAddBiasAddconv2d_26/Conv2D:output:0(conv2d_26/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_26/BiasAdd~
conv2d_26/TanhTanhconv2d_26/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
conv2d_26/Tanh?
IdentityIdentityconv2d_26/Tanh:y:0!^conv2d_24/BiasAdd/ReadVariableOp ^conv2d_24/Conv2D/ReadVariableOp!^conv2d_25/BiasAdd/ReadVariableOp ^conv2d_25/Conv2D/ReadVariableOp!^conv2d_26/BiasAdd/ReadVariableOp ^conv2d_26/Conv2D/ReadVariableOp^dense_4/BiasAdd/ReadVariableOp^dense_4/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*7
_input_shapes&
$:??????????: : : : : : : : 2D
 conv2d_24/BiasAdd/ReadVariableOp conv2d_24/BiasAdd/ReadVariableOp2B
conv2d_24/Conv2D/ReadVariableOpconv2d_24/Conv2D/ReadVariableOp2D
 conv2d_25/BiasAdd/ReadVariableOp conv2d_25/BiasAdd/ReadVariableOp2B
conv2d_25/Conv2D/ReadVariableOpconv2d_25/Conv2D/ReadVariableOp2D
 conv2d_26/BiasAdd/ReadVariableOp conv2d_26/BiasAdd/ReadVariableOp2B
conv2d_26/Conv2D/ReadVariableOpconv2d_26/Conv2D/ReadVariableOp2@
dense_4/BiasAdd/ReadVariableOpdense_4/BiasAdd/ReadVariableOp2>
dense_4/MatMul/ReadVariableOpdense_4/MatMul/ReadVariableOp:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107885292

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
E
)__inference_glu_4_layer_call_fn_107885302

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
D__inference_glu_4_layer_call_and_return_conditional_losses_1078843312
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
?
`
D__inference_glu_4_layer_call_and_return_conditional_losses_107884592

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
d
H__inference_reshape_4_layer_call_and_return_conditional_losses_107885336

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
?	
?
F__inference_dense_4_layer_call_and_return_conditional_losses_107884403

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
H__inference_conv2d_24_layer_call_and_return_conditional_losses_107884437

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
?
/__inference_generator_4_layer_call_fn_107885168

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
J__inference_generator_4_layer_call_and_return_conditional_losses_1078846742
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
_tf_keras_model?{"name": "generator_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Generator", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [128, 128]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Generator"}}
?

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*\&call_and_return_all_conditional_losses
]__call__"?
_tf_keras_layer?{"name": "dense_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_4", "trainable": true, "dtype": "float32", "units": 12544, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 128}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 128]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*^&call_and_return_all_conditional_losses
___call__"?
_tf_keras_model?{"name": "glu_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "GLU", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 12544]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "GLU"}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*`&call_and_return_all_conditional_losses
a__call__"?
_tf_keras_layer?{"name": "reshape_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Reshape", "config": {"name": "reshape_4", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 128]}}, "shared_object_id": 4}
?
trainable_variables
regularization_losses
	variables
	keras_api
*b&call_and_return_all_conditional_losses
c__call__"?
_tf_keras_layer?{"name": "up_sampling2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "UpSampling2D", "config": {"name": "up_sampling2d_4", "trainable": true, "dtype": "float32", "size": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last", "interpolation": "nearest"}, "shared_object_id": 5, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 6}}
?


kernel
 bias
!trainable_variables
"regularization_losses
#	variables
$	keras_api
*d&call_and_return_all_conditional_losses
e__call__"?	
_tf_keras_layer?	{"name": "conv2d_24", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_24", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 7}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 8}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 10}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 128]}}
?


%kernel
&bias
'trainable_variables
(regularization_losses
)	variables
*	keras_api
*f&call_and_return_all_conditional_losses
g__call__"?	
_tf_keras_layer?	{"name": "conv2d_25", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_25", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 14}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 64]}}
?


+kernel
,bias
-trainable_variables
.regularization_losses
/	variables
0	keras_api
*h&call_and_return_all_conditional_losses
i__call__"?	
_tf_keras_layer?	{"name": "conv2d_26", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_26", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "tanh", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 15}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 16}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 17, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
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
??b2dense_4/kernel
:?b2dense_4/bias
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
,:*??2conv2d_24/kernel
:?2conv2d_24/bias
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
*:(@@2conv2d_25/kernel
:@2conv2d_25/bias
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
*:( 2conv2d_26/kernel
:2conv2d_26/bias
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
J__inference_generator_4_layer_call_and_return_conditional_losses_107884874
J__inference_generator_4_layer_call_and_return_conditional_losses_107884951
J__inference_generator_4_layer_call_and_return_conditional_losses_107885028
J__inference_generator_4_layer_call_and_return_conditional_losses_107885105?
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
/__inference_generator_4_layer_call_fn_107885126
/__inference_generator_4_layer_call_fn_107885147
/__inference_generator_4_layer_call_fn_107885168
/__inference_generator_4_layer_call_fn_107885189?
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
$__inference__wrapped_model_107884289?
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
F__inference_dense_4_layer_call_and_return_conditional_losses_107885199?
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
+__inference_dense_4_layer_call_fn_107885208?
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
D__inference_glu_4_layer_call_and_return_conditional_losses_107885222
D__inference_glu_4_layer_call_and_return_conditional_losses_107885236
D__inference_glu_4_layer_call_and_return_conditional_losses_107884353
D__inference_glu_4_layer_call_and_return_conditional_losses_107884367
D__inference_glu_4_layer_call_and_return_conditional_losses_107885250
D__inference_glu_4_layer_call_and_return_conditional_losses_107885264
D__inference_glu_4_layer_call_and_return_conditional_losses_107885278
D__inference_glu_4_layer_call_and_return_conditional_losses_107885292?
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
)__inference_glu_4_layer_call_fn_107884310
)__inference_glu_4_layer_call_fn_107885297
)__inference_glu_4_layer_call_fn_107885302
)__inference_glu_4_layer_call_fn_107884339
)__inference_glu_4_layer_call_fn_107885307
)__inference_glu_4_layer_call_fn_107885312
)__inference_glu_4_layer_call_fn_107885317
)__inference_glu_4_layer_call_fn_107885322?
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
H__inference_reshape_4_layer_call_and_return_conditional_losses_107885336?
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
-__inference_reshape_4_layer_call_fn_107885341?
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
N__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_107884380?
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
3__inference_up_sampling2d_4_layer_call_fn_107884386?
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
H__inference_conv2d_24_layer_call_and_return_conditional_losses_107885351?
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
-__inference_conv2d_24_layer_call_fn_107885360?
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
H__inference_conv2d_25_layer_call_and_return_conditional_losses_107885370?
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
-__inference_conv2d_25_layer_call_fn_107885379?
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
H__inference_conv2d_26_layer_call_and_return_conditional_losses_107885390?
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
-__inference_conv2d_26_layer_call_fn_107885399?
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
'__inference_signature_wrapper_107884797input_1"?
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
$__inference__wrapped_model_107884289z %&+,1?.
'?$
"?
input_1??????????
? ";?8
6
output_1*?'
output_1??????????
H__inference_conv2d_24_layer_call_and_return_conditional_losses_107885351? J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
-__inference_conv2d_24_layer_call_fn_107885360? J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
H__inference_conv2d_25_layer_call_and_return_conditional_losses_107885370?%&I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+???????????????????????????@
? ?
-__inference_conv2d_25_layer_call_fn_107885379?%&I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+???????????????????????????@?
H__inference_conv2d_26_layer_call_and_return_conditional_losses_107885390?+,I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
-__inference_conv2d_26_layer_call_fn_107885399?+,I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
F__inference_dense_4_layer_call_and_return_conditional_losses_107885199^0?-
&?#
!?
inputs??????????
? "&?#
?
0??????????b
? ?
+__inference_dense_4_layer_call_fn_107885208Q0?-
&?#
!?
inputs??????????
? "???????????b?
J__inference_generator_4_layer_call_and_return_conditional_losses_107884874o %&+,4?1
*?'
!?
inputs??????????
p 
? "-?*
#? 
0?????????
? ?
J__inference_generator_4_layer_call_and_return_conditional_losses_107884951o %&+,4?1
*?'
!?
inputs??????????
p
? "-?*
#? 
0?????????
? ?
J__inference_generator_4_layer_call_and_return_conditional_losses_107885028p %&+,5?2
+?(
"?
input_1??????????
p 
? "-?*
#? 
0?????????
? ?
J__inference_generator_4_layer_call_and_return_conditional_losses_107885105p %&+,5?2
+?(
"?
input_1??????????
p
? "-?*
#? 
0?????????
? ?
/__inference_generator_4_layer_call_fn_107885126u %&+,5?2
+?(
"?
input_1??????????
p 
? "2?/+????????????????????????????
/__inference_generator_4_layer_call_fn_107885147t %&+,4?1
*?'
!?
inputs??????????
p 
? "2?/+????????????????????????????
/__inference_generator_4_layer_call_fn_107885168t %&+,4?1
*?'
!?
inputs??????????
p
? "2?/+????????????????????????????
/__inference_generator_4_layer_call_fn_107885189u %&+,5?2
+?(
"?
input_1??????????
p
? "2?/+????????????????????????????
D__inference_glu_4_layer_call_and_return_conditional_losses_107884353kA?>
'?$
"?
input_1??????????b
?

trainingp "&?#
?
0??????????1
? ?
D__inference_glu_4_layer_call_and_return_conditional_losses_107884367kA?>
'?$
"?
input_1??????????b
?

trainingp"&?#
?
0??????????1
? ?
D__inference_glu_4_layer_call_and_return_conditional_losses_107885222j@?=
&?#
!?
inputs??????????b
?

trainingp "&?#
?
0??????????1
? ?
D__inference_glu_4_layer_call_and_return_conditional_losses_107885236j@?=
&?#
!?
inputs??????????b
?

trainingp"&?#
?
0??????????1
? ?
D__inference_glu_4_layer_call_and_return_conditional_losses_107885250?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp "??<
5?2
0+???????????????????????????@
? ?
D__inference_glu_4_layer_call_and_return_conditional_losses_107885264?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp "??<
5?2
0+??????????????????????????? 
? ?
D__inference_glu_4_layer_call_and_return_conditional_losses_107885278?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp"??<
5?2
0+??????????????????????????? 
? ?
D__inference_glu_4_layer_call_and_return_conditional_losses_107885292?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp"??<
5?2
0+???????????????????????????@
? ?
)__inference_glu_4_layer_call_fn_107884310^A?>
'?$
"?
input_1??????????b
?

trainingp "???????????1?
)__inference_glu_4_layer_call_fn_107884339^A?>
'?$
"?
input_1??????????b
?

trainingp"???????????1?
)__inference_glu_4_layer_call_fn_107885297]@?=
&?#
!?
inputs??????????b
?

trainingp "???????????1?
)__inference_glu_4_layer_call_fn_107885302]@?=
&?#
!?
inputs??????????b
?

trainingp"???????????1?
)__inference_glu_4_layer_call_fn_107885307?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp "2?/+??????????????????????????? ?
)__inference_glu_4_layer_call_fn_107885312?Y?V
??<
:?7
inputs+???????????????????????????@
?

trainingp"2?/+??????????????????????????? ?
)__inference_glu_4_layer_call_fn_107885317?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp "2?/+???????????????????????????@?
)__inference_glu_4_layer_call_fn_107885322?Z?W
@?=
;?8
inputs,????????????????????????????
?

trainingp"2?/+???????????????????????????@?
H__inference_reshape_4_layer_call_and_return_conditional_losses_107885336b0?-
&?#
!?
inputs??????????1
? ".?+
$?!
0??????????
? ?
-__inference_reshape_4_layer_call_fn_107885341U0?-
&?#
!?
inputs??????????1
? "!????????????
'__inference_signature_wrapper_107884797? %&+,<?9
? 
2?/
-
input_1"?
input_1??????????";?8
6
output_1*?'
output_1??????????
N__inference_up_sampling2d_4_layer_call_and_return_conditional_losses_107884380?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
3__inference_up_sampling2d_4_layer_call_fn_107884386?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84????????????????????????????????????