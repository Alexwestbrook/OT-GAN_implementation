??
??
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
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
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
0
Neg
x"T
y"T"
Ttype:
2
	
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
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
H
ShardedFilename
basename	
shard

num_shards
filename
-
Sqrt
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
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
conv2d_27/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_27/kernel
}
$conv2d_27/kernel/Read/ReadVariableOpReadVariableOpconv2d_27/kernel*&
_output_shapes
: *
dtype0
t
conv2d_27/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_27/bias
m
"conv2d_27/bias/Read/ReadVariableOpReadVariableOpconv2d_27/bias*
_output_shapes
: *
dtype0
?
conv2d_28/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_28/kernel
}
$conv2d_28/kernel/Read/ReadVariableOpReadVariableOpconv2d_28/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_28/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_28/bias
m
"conv2d_28/bias/Read/ReadVariableOpReadVariableOpconv2d_28/bias*
_output_shapes
:@*
dtype0
?
conv2d_29/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_29/kernel

$conv2d_29/kernel/Read/ReadVariableOpReadVariableOpconv2d_29/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_29/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_29/bias
n
"conv2d_29/bias/Read/ReadVariableOpReadVariableOpconv2d_29/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
?
	conv1
	conv2
	conv3
flatten
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
h


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
R
trainable_variables
regularization_losses
	variables
	keras_api
*

0
1
2
3
4
5
 
*

0
1
2
3
4
5
?

 layers
trainable_variables
!non_trainable_variables
regularization_losses
"metrics
#layer_regularization_losses
$layer_metrics
	variables
 
MK
VARIABLE_VALUEconv2d_27/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_27/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
?

%layers
trainable_variables
&non_trainable_variables
regularization_losses
'metrics
(layer_regularization_losses
)layer_metrics
	variables
MK
VARIABLE_VALUEconv2d_28/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_28/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

*layers
trainable_variables
+non_trainable_variables
regularization_losses
,metrics
-layer_regularization_losses
.layer_metrics
	variables
MK
VARIABLE_VALUEconv2d_29/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_29/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

/layers
trainable_variables
0non_trainable_variables
regularization_losses
1metrics
2layer_regularization_losses
3layer_metrics
	variables
 
 
 
?

4layers
trainable_variables
5non_trainable_variables
regularization_losses
6metrics
7layer_regularization_losses
8layer_metrics
	variables

0
1
2
3
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
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *0
f+R)
'__inference_signature_wrapper_107885846
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_27/kernel/Read/ReadVariableOp"conv2d_27/bias/Read/ReadVariableOp$conv2d_28/kernel/Read/ReadVariableOp"conv2d_28/bias/Read/ReadVariableOp$conv2d_29/kernel/Read/ReadVariableOp"conv2d_29/bias/Read/ReadVariableOpConst*
Tin

2*
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
"__inference__traced_save_107886199
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_27/kernelconv2d_27/biasconv2d_28/kernelconv2d_28/biasconv2d_29/kernelconv2d_29/bias*
Tin
	2*
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
%__inference__traced_restore_107886227??
?
?
H__inference_conv2d_29_layer_call_and_return_conditional_losses_107885630

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
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAddj
	CRelu/NegNegBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
	CRelu/Negc

CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2

CRelu/axis?
CReluConcatV2BiasAdd:output:0CRelu/Neg:y:0CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
CReluk

CRelu/ReluReluCRelu:output:0*
T0*0
_output_shapes
:??????????2

CRelu/Relu?
IdentityIdentityCRelu/Relu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_29_layer_call_and_return_conditional_losses_107886138

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
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAddj
	CRelu/NegNegBiasAdd:output:0*
T0*0
_output_shapes
:??????????2
	CRelu/Negc

CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2

CRelu/axis?
CReluConcatV2BiasAdd:output:0CRelu/Neg:y:0CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
CReluk

CRelu/ReluReluCRelu:output:0*
T0*0
_output_shapes
:??????????2

CRelu/Relu?
IdentityIdentityCRelu/Relu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_4_layer_call_and_return_conditional_losses_107886153

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????b2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_27_layer_call_and_return_conditional_losses_107885590

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAddi
	CRelu/NegNegBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
	CRelu/Negc

CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2

CRelu/axis?
CReluConcatV2BiasAdd:output:0CRelu/Neg:y:0CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
CReluj

CRelu/ReluReluCRelu:output:0*
T0*/
_output_shapes
:?????????@2

CRelu/Relu?
IdentityIdentityCRelu/Relu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
-__inference_conv2d_28_layer_call_fn_107886124

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
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_28_layer_call_and_return_conditional_losses_1078856102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
G__inference_critic_4_layer_call_and_return_conditional_losses_107885650

inputs-
conv2d_27_107885591: !
conv2d_27_107885593: -
conv2d_28_107885611:@@!
conv2d_28_107885613:@/
conv2d_29_107885631:??"
conv2d_29_107885633:	?
identity??!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_107885591conv2d_27_107885593*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_27_layer_call_and_return_conditional_losses_1078855902#
!conv2d_27/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_107885611conv2d_28_107885613*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_28_layer_call_and_return_conditional_losses_1078856102#
!conv2d_28/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_107885631conv2d_29_107885633*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_29_layer_call_and_return_conditional_losses_1078856302#
!conv2d_29/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_1078856422
flatten_4/PartitionedCall?
norm/mulMul"flatten_4/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0*
T0*(
_output_shapes
:??????????b2

norm/mul?
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices?
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

norm/Sumc
	norm/SqrtSqrtnorm/Sum:output:0*
T0*'
_output_shapes
:?????????2
	norm/Sqrt?
truedivRealDiv"flatten_4/PartitionedCall:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
I
-__inference_flatten_4_layer_call_fn_107886158

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
:??????????b* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_1078856422
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?3
?
G__inference_critic_4_layer_call_and_return_conditional_losses_107886010
input_1B
(conv2d_27_conv2d_readvariableop_resource: 7
)conv2d_27_biasadd_readvariableop_resource: B
(conv2d_28_conv2d_readvariableop_resource:@@7
)conv2d_28_biasadd_readvariableop_resource:@D
(conv2d_29_conv2d_readvariableop_resource:??8
)conv2d_29_biasadd_readvariableop_resource:	?
identity?? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2Dinput_1'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/BiasAdd?
conv2d_27/CRelu/NegNegconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/CRelu/Negw
conv2d_27/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_27/CRelu/axis?
conv2d_27/CReluConcatV2conv2d_27/BiasAdd:output:0conv2d_27/CRelu/Neg:y:0conv2d_27/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_27/CRelu?
conv2d_27/CRelu/ReluReluconv2d_27/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_27/CRelu/Relu?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2D"conv2d_27/CRelu/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_28/BiasAdd?
conv2d_28/CRelu/NegNegconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_28/CRelu/Negw
conv2d_28/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_28/CRelu/axis?
conv2d_28/CReluConcatV2conv2d_28/BiasAdd:output:0conv2d_28/CRelu/Neg:y:0conv2d_28/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_28/CRelu?
conv2d_28/CRelu/ReluReluconv2d_28/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_28/CRelu/Relu?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2D"conv2d_28/CRelu/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_29/BiasAdd?
conv2d_29/CRelu/NegNegconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu/Negw
conv2d_29/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_29/CRelu/axis?
conv2d_29/CReluConcatV2conv2d_29/BiasAdd:output:0conv2d_29/CRelu/Neg:y:0conv2d_29/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu?
conv2d_29/CRelu/ReluReluconv2d_29/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_4/Const?
flatten_4/ReshapeReshape"conv2d_29/CRelu/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_4/Reshape?
norm/mulMulflatten_4/Reshape:output:0flatten_4/Reshape:output:0*
T0*(
_output_shapes
:??????????b2

norm/mul?
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices?
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

norm/Sumc
	norm/SqrtSqrtnorm/Sum:output:0*
T0*'
_output_shapes
:?????????2
	norm/Sqrt{
truedivRealDivflatten_4/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
H__inference_conv2d_28_layer_call_and_return_conditional_losses_107885610

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
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAddi
	CRelu/NegNegBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
	CRelu/Negc

CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2

CRelu/axis?
CReluConcatV2BiasAdd:output:0CRelu/Neg:y:0CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
CReluk

CRelu/ReluReluCRelu:output:0*
T0*0
_output_shapes
:??????????2

CRelu/Relu?
IdentityIdentityCRelu/Relu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
H__inference_conv2d_28_layer_call_and_return_conditional_losses_107886115

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
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAddi
	CRelu/NegNegBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
	CRelu/Negc

CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2

CRelu/axis?
CReluConcatV2BiasAdd:output:0CRelu/Neg:y:0CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
CReluk

CRelu/ReluReluCRelu:output:0*
T0*0
_output_shapes
:??????????2

CRelu/Relu?
IdentityIdentityCRelu/Relu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?3
?
G__inference_critic_4_layer_call_and_return_conditional_losses_107885969
input_1B
(conv2d_27_conv2d_readvariableop_resource: 7
)conv2d_27_biasadd_readvariableop_resource: B
(conv2d_28_conv2d_readvariableop_resource:@@7
)conv2d_28_biasadd_readvariableop_resource:@D
(conv2d_29_conv2d_readvariableop_resource:??8
)conv2d_29_biasadd_readvariableop_resource:	?
identity?? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2Dinput_1'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/BiasAdd?
conv2d_27/CRelu/NegNegconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/CRelu/Negw
conv2d_27/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_27/CRelu/axis?
conv2d_27/CReluConcatV2conv2d_27/BiasAdd:output:0conv2d_27/CRelu/Neg:y:0conv2d_27/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_27/CRelu?
conv2d_27/CRelu/ReluReluconv2d_27/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_27/CRelu/Relu?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2D"conv2d_27/CRelu/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_28/BiasAdd?
conv2d_28/CRelu/NegNegconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_28/CRelu/Negw
conv2d_28/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_28/CRelu/axis?
conv2d_28/CReluConcatV2conv2d_28/BiasAdd:output:0conv2d_28/CRelu/Neg:y:0conv2d_28/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_28/CRelu?
conv2d_28/CRelu/ReluReluconv2d_28/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_28/CRelu/Relu?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2D"conv2d_28/CRelu/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_29/BiasAdd?
conv2d_29/CRelu/NegNegconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu/Negw
conv2d_29/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_29/CRelu/axis?
conv2d_29/CReluConcatV2conv2d_29/BiasAdd:output:0conv2d_29/CRelu/Neg:y:0conv2d_29/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu?
conv2d_29/CRelu/ReluReluconv2d_29/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_4/Const?
flatten_4/ReshapeReshape"conv2d_29/CRelu/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_4/Reshape?
norm/mulMulflatten_4/Reshape:output:0flatten_4/Reshape:output:0*
T0*(
_output_shapes
:??????????b2

norm/mul?
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices?
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

norm/Sumc
	norm/SqrtSqrtnorm/Sum:output:0*
T0*'
_output_shapes
:?????????2
	norm/Sqrt{
truedivRealDivflatten_4/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
'__inference_signature_wrapper_107885846
input_1!
unknown: 
	unknown_0: #
	unknown_1:@@
	unknown_2:@%
	unknown_3:??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *-
f(R&
$__inference__wrapped_model_1078855692
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
-__inference_conv2d_29_layer_call_fn_107886147

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
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_29_layer_call_and_return_conditional_losses_1078856302
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
-__inference_conv2d_27_layer_call_fn_107886101

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_27_layer_call_and_return_conditional_losses_1078855902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
d
H__inference_flatten_4_layer_call_and_return_conditional_losses_107885642

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????b2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*/
_input_shapes
:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_critic_4_layer_call_fn_107886027
input_1!
unknown: 
	unknown_0: #
	unknown_1:@@
	unknown_2:@%
	unknown_3:??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_critic_4_layer_call_and_return_conditional_losses_1078856502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
"__inference__traced_save_107886199
file_prefix/
+savev2_conv2d_27_kernel_read_readvariableop-
)savev2_conv2d_27_bias_read_readvariableop/
+savev2_conv2d_28_kernel_read_readvariableop-
)savev2_conv2d_28_bias_read_readvariableop/
+savev2_conv2d_29_kernel_read_readvariableop-
)savev2_conv2d_29_bias_read_readvariableop
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
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_27_kernel_read_readvariableop)savev2_conv2d_27_bias_read_readvariableop+savev2_conv2d_28_kernel_read_readvariableop)savev2_conv2d_28_bias_read_readvariableop+savev2_conv2d_29_kernel_read_readvariableop)savev2_conv2d_29_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
	22
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

identity_1Identity_1:output:0*b
_input_shapesQ
O: : : :@@:@:??:?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
:@@: 

_output_shapes
:@:.*
(
_output_shapes
:??:!

_output_shapes	
:?:

_output_shapes
: 
?3
?
G__inference_critic_4_layer_call_and_return_conditional_losses_107885928

inputsB
(conv2d_27_conv2d_readvariableop_resource: 7
)conv2d_27_biasadd_readvariableop_resource: B
(conv2d_28_conv2d_readvariableop_resource:@@7
)conv2d_28_biasadd_readvariableop_resource:@D
(conv2d_29_conv2d_readvariableop_resource:??8
)conv2d_29_biasadd_readvariableop_resource:	?
identity?? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/BiasAdd?
conv2d_27/CRelu/NegNegconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/CRelu/Negw
conv2d_27/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_27/CRelu/axis?
conv2d_27/CReluConcatV2conv2d_27/BiasAdd:output:0conv2d_27/CRelu/Neg:y:0conv2d_27/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_27/CRelu?
conv2d_27/CRelu/ReluReluconv2d_27/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_27/CRelu/Relu?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2D"conv2d_27/CRelu/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_28/BiasAdd?
conv2d_28/CRelu/NegNegconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_28/CRelu/Negw
conv2d_28/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_28/CRelu/axis?
conv2d_28/CReluConcatV2conv2d_28/BiasAdd:output:0conv2d_28/CRelu/Neg:y:0conv2d_28/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_28/CRelu?
conv2d_28/CRelu/ReluReluconv2d_28/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_28/CRelu/Relu?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2D"conv2d_28/CRelu/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_29/BiasAdd?
conv2d_29/CRelu/NegNegconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu/Negw
conv2d_29/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_29/CRelu/axis?
conv2d_29/CReluConcatV2conv2d_29/BiasAdd:output:0conv2d_29/CRelu/Neg:y:0conv2d_29/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu?
conv2d_29/CRelu/ReluReluconv2d_29/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_4/Const?
flatten_4/ReshapeReshape"conv2d_29/CRelu/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_4/Reshape?
norm/mulMulflatten_4/Reshape:output:0flatten_4/Reshape:output:0*
T0*(
_output_shapes
:??????????b2

norm/mul?
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices?
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

norm/Sumc
	norm/SqrtSqrtnorm/Sum:output:0*
T0*'
_output_shapes
:?????????2
	norm/Sqrt{
truedivRealDivflatten_4/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?3
?
G__inference_critic_4_layer_call_and_return_conditional_losses_107885887

inputsB
(conv2d_27_conv2d_readvariableop_resource: 7
)conv2d_27_biasadd_readvariableop_resource: B
(conv2d_28_conv2d_readvariableop_resource:@@7
)conv2d_28_biasadd_readvariableop_resource:@D
(conv2d_29_conv2d_readvariableop_resource:??8
)conv2d_29_biasadd_readvariableop_resource:	?
identity?? conv2d_27/BiasAdd/ReadVariableOp?conv2d_27/Conv2D/ReadVariableOp? conv2d_28/BiasAdd/ReadVariableOp?conv2d_28/Conv2D/ReadVariableOp? conv2d_29/BiasAdd/ReadVariableOp?conv2d_29/Conv2D/ReadVariableOp?
conv2d_27/Conv2D/ReadVariableOpReadVariableOp(conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_27/Conv2D/ReadVariableOp?
conv2d_27/Conv2DConv2Dinputs'conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_27/Conv2D?
 conv2d_27/BiasAdd/ReadVariableOpReadVariableOp)conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_27/BiasAdd/ReadVariableOp?
conv2d_27/BiasAddBiasAddconv2d_27/Conv2D:output:0(conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/BiasAdd?
conv2d_27/CRelu/NegNegconv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_27/CRelu/Negw
conv2d_27/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_27/CRelu/axis?
conv2d_27/CReluConcatV2conv2d_27/BiasAdd:output:0conv2d_27/CRelu/Neg:y:0conv2d_27/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_27/CRelu?
conv2d_27/CRelu/ReluReluconv2d_27/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_27/CRelu/Relu?
conv2d_28/Conv2D/ReadVariableOpReadVariableOp(conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_28/Conv2D/ReadVariableOp?
conv2d_28/Conv2DConv2D"conv2d_27/CRelu/Relu:activations:0'conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_28/Conv2D?
 conv2d_28/BiasAdd/ReadVariableOpReadVariableOp)conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_28/BiasAdd/ReadVariableOp?
conv2d_28/BiasAddBiasAddconv2d_28/Conv2D:output:0(conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_28/BiasAdd?
conv2d_28/CRelu/NegNegconv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_28/CRelu/Negw
conv2d_28/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_28/CRelu/axis?
conv2d_28/CReluConcatV2conv2d_28/BiasAdd:output:0conv2d_28/CRelu/Neg:y:0conv2d_28/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_28/CRelu?
conv2d_28/CRelu/ReluReluconv2d_28/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_28/CRelu/Relu?
conv2d_29/Conv2D/ReadVariableOpReadVariableOp(conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_29/Conv2D/ReadVariableOp?
conv2d_29/Conv2DConv2D"conv2d_28/CRelu/Relu:activations:0'conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_29/Conv2D?
 conv2d_29/BiasAdd/ReadVariableOpReadVariableOp)conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_29/BiasAdd/ReadVariableOp?
conv2d_29/BiasAddBiasAddconv2d_29/Conv2D:output:0(conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_29/BiasAdd?
conv2d_29/CRelu/NegNegconv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu/Negw
conv2d_29/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_29/CRelu/axis?
conv2d_29/CReluConcatV2conv2d_29/BiasAdd:output:0conv2d_29/CRelu/Neg:y:0conv2d_29/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu?
conv2d_29/CRelu/ReluReluconv2d_29/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_29/CRelu/Relus
flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_4/Const?
flatten_4/ReshapeReshape"conv2d_29/CRelu/Relu:activations:0flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_4/Reshape?
norm/mulMulflatten_4/Reshape:output:0flatten_4/Reshape:output:0*
T0*(
_output_shapes
:??????????b2

norm/mul?
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices?
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

norm/Sumc
	norm/SqrtSqrtnorm/Sum:output:0*
T0*'
_output_shapes
:?????????2
	norm/Sqrt{
truedivRealDivflatten_4/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_27/BiasAdd/ReadVariableOp ^conv2d_27/Conv2D/ReadVariableOp!^conv2d_28/BiasAdd/ReadVariableOp ^conv2d_28/Conv2D/ReadVariableOp!^conv2d_29/BiasAdd/ReadVariableOp ^conv2d_29/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_27/BiasAdd/ReadVariableOp conv2d_27/BiasAdd/ReadVariableOp2B
conv2d_27/Conv2D/ReadVariableOpconv2d_27/Conv2D/ReadVariableOp2D
 conv2d_28/BiasAdd/ReadVariableOp conv2d_28/BiasAdd/ReadVariableOp2B
conv2d_28/Conv2D/ReadVariableOpconv2d_28/Conv2D/ReadVariableOp2D
 conv2d_29/BiasAdd/ReadVariableOp conv2d_29/BiasAdd/ReadVariableOp2B
conv2d_29/Conv2D/ReadVariableOpconv2d_29/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_critic_4_layer_call_fn_107886061

inputs!
unknown: 
	unknown_0: #
	unknown_1:@@
	unknown_2:@%
	unknown_3:??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_critic_4_layer_call_and_return_conditional_losses_1078857452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
%__inference__traced_restore_107886227
file_prefix;
!assignvariableop_conv2d_27_kernel: /
!assignvariableop_1_conv2d_27_bias: =
#assignvariableop_2_conv2d_28_kernel:@@/
!assignvariableop_3_conv2d_28_bias:@?
#assignvariableop_4_conv2d_29_kernel:??0
!assignvariableop_5_conv2d_29_bias:	?

identity_7??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?
value?B?B'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv1/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv2/bias/.ATTRIBUTES/VARIABLE_VALUEB'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUEB%conv3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*!
valueBB B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*0
_output_shapes
:::::::*
dtypes
	22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_27_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_27_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_28_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_28_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_29_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_29_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_59
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_6Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_6?

Identity_7IdentityIdentity_6:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5*
T0*
_output_shapes
: 2

Identity_7"!

identity_7Identity_7:output:0*!
_input_shapes
: : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_5:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
?
G__inference_critic_4_layer_call_and_return_conditional_losses_107885745

inputs-
conv2d_27_107885723: !
conv2d_27_107885725: -
conv2d_28_107885728:@@!
conv2d_28_107885730:@/
conv2d_29_107885733:??"
conv2d_29_107885735:	?
identity??!conv2d_27/StatefulPartitionedCall?!conv2d_28/StatefulPartitionedCall?!conv2d_29/StatefulPartitionedCall?
!conv2d_27/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_27_107885723conv2d_27_107885725*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_27_layer_call_and_return_conditional_losses_1078855902#
!conv2d_27/StatefulPartitionedCall?
!conv2d_28/StatefulPartitionedCallStatefulPartitionedCall*conv2d_27/StatefulPartitionedCall:output:0conv2d_28_107885728conv2d_28_107885730*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_28_layer_call_and_return_conditional_losses_1078856102#
!conv2d_28/StatefulPartitionedCall?
!conv2d_29/StatefulPartitionedCallStatefulPartitionedCall*conv2d_28/StatefulPartitionedCall:output:0conv2d_29_107885733conv2d_29_107885735*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_conv2d_29_layer_call_and_return_conditional_losses_1078856302#
!conv2d_29/StatefulPartitionedCall?
flatten_4/PartitionedCallPartitionedCall*conv2d_29/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b* 
_read_only_resource_inputs
 *2
config_proto" 

CPU

GPU2*0,1J 8? *Q
fLRJ
H__inference_flatten_4_layer_call_and_return_conditional_losses_1078856422
flatten_4/PartitionedCall?
norm/mulMul"flatten_4/PartitionedCall:output:0"flatten_4/PartitionedCall:output:0*
T0*(
_output_shapes
:??????????b2

norm/mul?
norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2
norm/Sum/reduction_indices?
norm/SumSumnorm/mul:z:0#norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2

norm/Sumc
	norm/SqrtSqrtnorm/Sum:output:0*
T0*'
_output_shapes
:?????????2
	norm/Sqrt?
truedivRealDiv"flatten_4/PartitionedCall:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0"^conv2d_27/StatefulPartitionedCall"^conv2d_28/StatefulPartitionedCall"^conv2d_29/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2F
!conv2d_27/StatefulPartitionedCall!conv2d_27/StatefulPartitionedCall2F
!conv2d_28/StatefulPartitionedCall!conv2d_28/StatefulPartitionedCall2F
!conv2d_29/StatefulPartitionedCall!conv2d_29/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
H__inference_conv2d_27_layer_call_and_return_conditional_losses_107886092

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2	
BiasAddi
	CRelu/NegNegBiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
	CRelu/Negc

CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2

CRelu/axis?
CReluConcatV2BiasAdd:output:0CRelu/Neg:y:0CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
CReluj

CRelu/ReluReluCRelu:output:0*
T0*/
_output_shapes
:?????????@2

CRelu/Relu?
IdentityIdentityCRelu/Relu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
,__inference_critic_4_layer_call_fn_107886044

inputs!
unknown: 
	unknown_0: #
	unknown_1:@@
	unknown_2:@%
	unknown_3:??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_critic_4_layer_call_and_return_conditional_losses_1078856502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
$__inference__wrapped_model_107885569
input_1K
1critic_4_conv2d_27_conv2d_readvariableop_resource: @
2critic_4_conv2d_27_biasadd_readvariableop_resource: K
1critic_4_conv2d_28_conv2d_readvariableop_resource:@@@
2critic_4_conv2d_28_biasadd_readvariableop_resource:@M
1critic_4_conv2d_29_conv2d_readvariableop_resource:??A
2critic_4_conv2d_29_biasadd_readvariableop_resource:	?
identity??)critic_4/conv2d_27/BiasAdd/ReadVariableOp?(critic_4/conv2d_27/Conv2D/ReadVariableOp?)critic_4/conv2d_28/BiasAdd/ReadVariableOp?(critic_4/conv2d_28/Conv2D/ReadVariableOp?)critic_4/conv2d_29/BiasAdd/ReadVariableOp?(critic_4/conv2d_29/Conv2D/ReadVariableOp?
(critic_4/conv2d_27/Conv2D/ReadVariableOpReadVariableOp1critic_4_conv2d_27_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(critic_4/conv2d_27/Conv2D/ReadVariableOp?
critic_4/conv2d_27/Conv2DConv2Dinput_10critic_4/conv2d_27/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
critic_4/conv2d_27/Conv2D?
)critic_4/conv2d_27/BiasAdd/ReadVariableOpReadVariableOp2critic_4_conv2d_27_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)critic_4/conv2d_27/BiasAdd/ReadVariableOp?
critic_4/conv2d_27/BiasAddBiasAdd"critic_4/conv2d_27/Conv2D:output:01critic_4/conv2d_27/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
critic_4/conv2d_27/BiasAdd?
critic_4/conv2d_27/CRelu/NegNeg#critic_4/conv2d_27/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
critic_4/conv2d_27/CRelu/Neg?
critic_4/conv2d_27/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
critic_4/conv2d_27/CRelu/axis?
critic_4/conv2d_27/CReluConcatV2#critic_4/conv2d_27/BiasAdd:output:0 critic_4/conv2d_27/CRelu/Neg:y:0&critic_4/conv2d_27/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
critic_4/conv2d_27/CRelu?
critic_4/conv2d_27/CRelu/ReluRelu!critic_4/conv2d_27/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
critic_4/conv2d_27/CRelu/Relu?
(critic_4/conv2d_28/Conv2D/ReadVariableOpReadVariableOp1critic_4_conv2d_28_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(critic_4/conv2d_28/Conv2D/ReadVariableOp?
critic_4/conv2d_28/Conv2DConv2D+critic_4/conv2d_27/CRelu/Relu:activations:00critic_4/conv2d_28/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
critic_4/conv2d_28/Conv2D?
)critic_4/conv2d_28/BiasAdd/ReadVariableOpReadVariableOp2critic_4_conv2d_28_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)critic_4/conv2d_28/BiasAdd/ReadVariableOp?
critic_4/conv2d_28/BiasAddBiasAdd"critic_4/conv2d_28/Conv2D:output:01critic_4/conv2d_28/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
critic_4/conv2d_28/BiasAdd?
critic_4/conv2d_28/CRelu/NegNeg#critic_4/conv2d_28/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
critic_4/conv2d_28/CRelu/Neg?
critic_4/conv2d_28/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
critic_4/conv2d_28/CRelu/axis?
critic_4/conv2d_28/CReluConcatV2#critic_4/conv2d_28/BiasAdd:output:0 critic_4/conv2d_28/CRelu/Neg:y:0&critic_4/conv2d_28/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
critic_4/conv2d_28/CRelu?
critic_4/conv2d_28/CRelu/ReluRelu!critic_4/conv2d_28/CRelu:output:0*
T0*0
_output_shapes
:??????????2
critic_4/conv2d_28/CRelu/Relu?
(critic_4/conv2d_29/Conv2D/ReadVariableOpReadVariableOp1critic_4_conv2d_29_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(critic_4/conv2d_29/Conv2D/ReadVariableOp?
critic_4/conv2d_29/Conv2DConv2D+critic_4/conv2d_28/CRelu/Relu:activations:00critic_4/conv2d_29/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
critic_4/conv2d_29/Conv2D?
)critic_4/conv2d_29/BiasAdd/ReadVariableOpReadVariableOp2critic_4_conv2d_29_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)critic_4/conv2d_29/BiasAdd/ReadVariableOp?
critic_4/conv2d_29/BiasAddBiasAdd"critic_4/conv2d_29/Conv2D:output:01critic_4/conv2d_29/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
critic_4/conv2d_29/BiasAdd?
critic_4/conv2d_29/CRelu/NegNeg#critic_4/conv2d_29/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
critic_4/conv2d_29/CRelu/Neg?
critic_4/conv2d_29/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
critic_4/conv2d_29/CRelu/axis?
critic_4/conv2d_29/CReluConcatV2#critic_4/conv2d_29/BiasAdd:output:0 critic_4/conv2d_29/CRelu/Neg:y:0&critic_4/conv2d_29/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
critic_4/conv2d_29/CRelu?
critic_4/conv2d_29/CRelu/ReluRelu!critic_4/conv2d_29/CRelu:output:0*
T0*0
_output_shapes
:??????????2
critic_4/conv2d_29/CRelu/Relu?
critic_4/flatten_4/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
critic_4/flatten_4/Const?
critic_4/flatten_4/ReshapeReshape+critic_4/conv2d_29/CRelu/Relu:activations:0!critic_4/flatten_4/Const:output:0*
T0*(
_output_shapes
:??????????b2
critic_4/flatten_4/Reshape?
critic_4/norm/mulMul#critic_4/flatten_4/Reshape:output:0#critic_4/flatten_4/Reshape:output:0*
T0*(
_output_shapes
:??????????b2
critic_4/norm/mul?
#critic_4/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2%
#critic_4/norm/Sum/reduction_indices?
critic_4/norm/SumSumcritic_4/norm/mul:z:0,critic_4/norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
critic_4/norm/Sum~
critic_4/norm/SqrtSqrtcritic_4/norm/Sum:output:0*
T0*'
_output_shapes
:?????????2
critic_4/norm/Sqrt?
critic_4/truedivRealDiv#critic_4/flatten_4/Reshape:output:0critic_4/norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2
critic_4/truediv?
IdentityIdentitycritic_4/truediv:z:0*^critic_4/conv2d_27/BiasAdd/ReadVariableOp)^critic_4/conv2d_27/Conv2D/ReadVariableOp*^critic_4/conv2d_28/BiasAdd/ReadVariableOp)^critic_4/conv2d_28/Conv2D/ReadVariableOp*^critic_4/conv2d_29/BiasAdd/ReadVariableOp)^critic_4/conv2d_29/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2V
)critic_4/conv2d_27/BiasAdd/ReadVariableOp)critic_4/conv2d_27/BiasAdd/ReadVariableOp2T
(critic_4/conv2d_27/Conv2D/ReadVariableOp(critic_4/conv2d_27/Conv2D/ReadVariableOp2V
)critic_4/conv2d_28/BiasAdd/ReadVariableOp)critic_4/conv2d_28/BiasAdd/ReadVariableOp2T
(critic_4/conv2d_28/Conv2D/ReadVariableOp(critic_4/conv2d_28/Conv2D/ReadVariableOp2V
)critic_4/conv2d_29/BiasAdd/ReadVariableOp)critic_4/conv2d_29/BiasAdd/ReadVariableOp2T
(critic_4/conv2d_29/Conv2D/ReadVariableOp(critic_4/conv2d_29/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
,__inference_critic_4_layer_call_fn_107886078
input_1!
unknown: 
	unknown_0: #
	unknown_1:@@
	unknown_2:@%
	unknown_3:??
	unknown_4:	?
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????b*(
_read_only_resource_inputs

*2
config_proto" 

CPU

GPU2*0,1J 8? *P
fKRI
G__inference_critic_4_layer_call_and_return_conditional_losses_1078857452
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????=
output_11
StatefulPartitionedCall:0??????????btensorflow/serving/predict:?u
?
	conv1
	conv2
	conv3
flatten
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
*9&call_and_return_all_conditional_losses
:__call__
;_default_save_signature"?
_tf_keras_model?{"name": "critic_4", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Critic", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 28, 28, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Critic"}}
?



kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*<&call_and_return_all_conditional_losses
=__call__"?	
_tf_keras_layer?	{"name": "conv2d_27", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_27", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "crelu_v2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*>&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_28", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_28", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "crelu_v2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 64]}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"?	
_tf_keras_layer?	{"name": "conv2d_29", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_29", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "crelu_v2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 128]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*B&call_and_return_all_conditional_losses
C__call__"?
_tf_keras_layer?{"name": "flatten_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_4", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 13}}
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
J

0
1
2
3
4
5"
trackable_list_wrapper
?

 layers
trainable_variables
!non_trainable_variables
regularization_losses
"metrics
#layer_regularization_losses
$layer_metrics
	variables
:__call__
;_default_save_signature
*9&call_and_return_all_conditional_losses
&9"call_and_return_conditional_losses"
_generic_user_object
,
Dserving_default"
signature_map
*:( 2conv2d_27/kernel
: 2conv2d_27/bias
.

0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
?

%layers
trainable_variables
&non_trainable_variables
regularization_losses
'metrics
(layer_regularization_losses
)layer_metrics
	variables
=__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
*:(@@2conv2d_28/kernel
:@2conv2d_28/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

*layers
trainable_variables
+non_trainable_variables
regularization_losses
,metrics
-layer_regularization_losses
.layer_metrics
	variables
?__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
,:*??2conv2d_29/kernel
:?2conv2d_29/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?

/layers
trainable_variables
0non_trainable_variables
regularization_losses
1metrics
2layer_regularization_losses
3layer_metrics
	variables
A__call__
*@&call_and_return_all_conditional_losses
&@"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

4layers
trainable_variables
5non_trainable_variables
regularization_losses
6metrics
7layer_regularization_losses
8layer_metrics
	variables
C__call__
*B&call_and_return_all_conditional_losses
&B"call_and_return_conditional_losses"
_generic_user_object
<
0
1
2
3"
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
G__inference_critic_4_layer_call_and_return_conditional_losses_107885887
G__inference_critic_4_layer_call_and_return_conditional_losses_107885928
G__inference_critic_4_layer_call_and_return_conditional_losses_107885969
G__inference_critic_4_layer_call_and_return_conditional_losses_107886010?
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
,__inference_critic_4_layer_call_fn_107886027
,__inference_critic_4_layer_call_fn_107886044
,__inference_critic_4_layer_call_fn_107886061
,__inference_critic_4_layer_call_fn_107886078?
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
$__inference__wrapped_model_107885569?
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
annotations? *.?+
)?&
input_1?????????
?2?
H__inference_conv2d_27_layer_call_and_return_conditional_losses_107886092?
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
-__inference_conv2d_27_layer_call_fn_107886101?
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
H__inference_conv2d_28_layer_call_and_return_conditional_losses_107886115?
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
-__inference_conv2d_28_layer_call_fn_107886124?
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
H__inference_conv2d_29_layer_call_and_return_conditional_losses_107886138?
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
-__inference_conv2d_29_layer_call_fn_107886147?
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
H__inference_flatten_4_layer_call_and_return_conditional_losses_107886153?
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
-__inference_flatten_4_layer_call_fn_107886158?
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
'__inference_signature_wrapper_107885846input_1"?
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
$__inference__wrapped_model_107885569x
8?5
.?+
)?&
input_1?????????
? "4?1
/
output_1#? 
output_1??????????b?
H__inference_conv2d_27_layer_call_and_return_conditional_losses_107886092l
7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????@
? ?
-__inference_conv2d_27_layer_call_fn_107886101_
7?4
-?*
(?%
inputs?????????
? " ??????????@?
H__inference_conv2d_28_layer_call_and_return_conditional_losses_107886115m7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
-__inference_conv2d_28_layer_call_fn_107886124`7?4
-?*
(?%
inputs?????????@
? "!????????????
H__inference_conv2d_29_layer_call_and_return_conditional_losses_107886138n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
-__inference_conv2d_29_layer_call_fn_107886147a8?5
.?+
)?&
inputs??????????
? "!????????????
G__inference_critic_4_layer_call_and_return_conditional_losses_107885887m
;?8
1?.
(?%
inputs?????????
p 
? "&?#
?
0??????????b
? ?
G__inference_critic_4_layer_call_and_return_conditional_losses_107885928m
;?8
1?.
(?%
inputs?????????
p
? "&?#
?
0??????????b
? ?
G__inference_critic_4_layer_call_and_return_conditional_losses_107885969n
<?9
2?/
)?&
input_1?????????
p 
? "&?#
?
0??????????b
? ?
G__inference_critic_4_layer_call_and_return_conditional_losses_107886010n
<?9
2?/
)?&
input_1?????????
p
? "&?#
?
0??????????b
? ?
,__inference_critic_4_layer_call_fn_107886027a
<?9
2?/
)?&
input_1?????????
p 
? "???????????b?
,__inference_critic_4_layer_call_fn_107886044`
;?8
1?.
(?%
inputs?????????
p 
? "???????????b?
,__inference_critic_4_layer_call_fn_107886061`
;?8
1?.
(?%
inputs?????????
p
? "???????????b?
,__inference_critic_4_layer_call_fn_107886078a
<?9
2?/
)?&
input_1?????????
p
? "???????????b?
H__inference_flatten_4_layer_call_and_return_conditional_losses_107886153b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????b
? ?
-__inference_flatten_4_layer_call_fn_107886158U8?5
.?+
)?&
inputs??????????
? "???????????b?
'__inference_signature_wrapper_107885846?
C?@
? 
9?6
4
input_1)?&
input_1?????????"4?1
/
output_1#? 
output_1??????????b