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
conv2d_21/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv2d_21/kernel
}
$conv2d_21/kernel/Read/ReadVariableOpReadVariableOpconv2d_21/kernel*&
_output_shapes
: *
dtype0
t
conv2d_21/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_21/bias
m
"conv2d_21/bias/Read/ReadVariableOpReadVariableOpconv2d_21/bias*
_output_shapes
: *
dtype0
?
conv2d_22/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*!
shared_nameconv2d_22/kernel
}
$conv2d_22/kernel/Read/ReadVariableOpReadVariableOpconv2d_22/kernel*&
_output_shapes
:@@*
dtype0
t
conv2d_22/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_22/bias
m
"conv2d_22/bias/Read/ReadVariableOpReadVariableOpconv2d_22/bias*
_output_shapes
:@*
dtype0
?
conv2d_23/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*!
shared_nameconv2d_23/kernel

$conv2d_23/kernel/Read/ReadVariableOpReadVariableOpconv2d_23/kernel*(
_output_shapes
:??*
dtype0
u
conv2d_23/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_23/bias
n
"conv2d_23/bias/Read/ReadVariableOpReadVariableOpconv2d_23/bias*
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
VARIABLE_VALUEconv2d_21/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_21/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_22/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_22/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE
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
VARIABLE_VALUEconv2d_23/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d_23/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/bias*
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
GPU2*0,1J 8? */
f*R(
&__inference_signature_wrapper_44800001
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename$conv2d_21/kernel/Read/ReadVariableOp"conv2d_21/bias/Read/ReadVariableOp$conv2d_22/kernel/Read/ReadVariableOp"conv2d_22/bias/Read/ReadVariableOp$conv2d_23/kernel/Read/ReadVariableOp"conv2d_23/bias/Read/ReadVariableOpConst*
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
GPU2*0,1J 8? **
f%R#
!__inference__traced_save_44800354
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_21/kernelconv2d_21/biasconv2d_22/kernelconv2d_22/biasconv2d_23/kernelconv2d_23/bias*
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
GPU2*0,1J 8? *-
f(R&
$__inference__traced_restore_44800382??
?
?
G__inference_conv2d_21_layer_call_and_return_conditional_losses_44799745

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
+__inference_critic_3_layer_call_fn_44800199

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
GPU2*0,1J 8? *O
fJRH
F__inference_critic_3_layer_call_and_return_conditional_losses_447998052
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
?
?
G__inference_conv2d_21_layer_call_and_return_conditional_losses_44800247

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
,__inference_conv2d_21_layer_call_fn_44800256

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
GPU2*0,1J 8? *P
fKRI
G__inference_conv2d_21_layer_call_and_return_conditional_losses_447997452
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
?
?
&__inference_signature_wrapper_44800001
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
GPU2*0,1J 8? *,
f'R%
#__inference__wrapped_model_447997242
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
?
?
+__inference_critic_3_layer_call_fn_44800216

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
GPU2*0,1J 8? *O
fJRH
F__inference_critic_3_layer_call_and_return_conditional_losses_447999002
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
?
?
+__inference_critic_3_layer_call_fn_44800233
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
GPU2*0,1J 8? *O
fJRH
F__inference_critic_3_layer_call_and_return_conditional_losses_447999002
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
!__inference__traced_save_44800354
file_prefix/
+savev2_conv2d_21_kernel_read_readvariableop-
)savev2_conv2d_21_bias_read_readvariableop/
+savev2_conv2d_22_kernel_read_readvariableop-
)savev2_conv2d_22_bias_read_readvariableop/
+savev2_conv2d_23_kernel_read_readvariableop-
)savev2_conv2d_23_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0+savev2_conv2d_21_kernel_read_readvariableop)savev2_conv2d_21_bias_read_readvariableop+savev2_conv2d_22_kernel_read_readvariableop)savev2_conv2d_22_bias_read_readvariableop+savev2_conv2d_23_kernel_read_readvariableop)savev2_conv2d_23_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
F__inference_critic_3_layer_call_and_return_conditional_losses_44799805

inputs,
conv2d_21_44799746:  
conv2d_21_44799748: ,
conv2d_22_44799766:@@ 
conv2d_22_44799768:@.
conv2d_23_44799786:??!
conv2d_23_44799788:	?
identity??!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_21_44799746conv2d_21_44799748*
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
GPU2*0,1J 8? *P
fKRI
G__inference_conv2d_21_layer_call_and_return_conditional_losses_447997452#
!conv2d_21/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_44799766conv2d_22_44799768*
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
GPU2*0,1J 8? *P
fKRI
G__inference_conv2d_22_layer_call_and_return_conditional_losses_447997652#
!conv2d_22/StatefulPartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_44799786conv2d_23_44799788*
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
GPU2*0,1J 8? *P
fKRI
G__inference_conv2d_23_layer_call_and_return_conditional_losses_447997852#
!conv2d_23/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_447997972
flatten_3/PartitionedCall?
norm/mulMul"flatten_3/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
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
truedivRealDiv"flatten_3/PartitionedCall:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_23_layer_call_and_return_conditional_losses_44800293

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
?
?
+__inference_critic_3_layer_call_fn_44800182
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
GPU2*0,1J 8? *O
fJRH
F__inference_critic_3_layer_call_and_return_conditional_losses_447998052
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
?
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_44800308

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
?
?
F__inference_critic_3_layer_call_and_return_conditional_losses_44799900

inputs,
conv2d_21_44799878:  
conv2d_21_44799880: ,
conv2d_22_44799883:@@ 
conv2d_22_44799885:@.
conv2d_23_44799888:??!
conv2d_23_44799890:	?
identity??!conv2d_21/StatefulPartitionedCall?!conv2d_22/StatefulPartitionedCall?!conv2d_23/StatefulPartitionedCall?
!conv2d_21/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_21_44799878conv2d_21_44799880*
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
GPU2*0,1J 8? *P
fKRI
G__inference_conv2d_21_layer_call_and_return_conditional_losses_447997452#
!conv2d_21/StatefulPartitionedCall?
!conv2d_22/StatefulPartitionedCallStatefulPartitionedCall*conv2d_21/StatefulPartitionedCall:output:0conv2d_22_44799883conv2d_22_44799885*
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
GPU2*0,1J 8? *P
fKRI
G__inference_conv2d_22_layer_call_and_return_conditional_losses_447997652#
!conv2d_22/StatefulPartitionedCall?
!conv2d_23/StatefulPartitionedCallStatefulPartitionedCall*conv2d_22/StatefulPartitionedCall:output:0conv2d_23_44799888conv2d_23_44799890*
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
GPU2*0,1J 8? *P
fKRI
G__inference_conv2d_23_layer_call_and_return_conditional_losses_447997852#
!conv2d_23/StatefulPartitionedCall?
flatten_3/PartitionedCallPartitionedCall*conv2d_23/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_447997972
flatten_3/PartitionedCall?
norm/mulMul"flatten_3/PartitionedCall:output:0"flatten_3/PartitionedCall:output:0*
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
truedivRealDiv"flatten_3/PartitionedCall:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0"^conv2d_21/StatefulPartitionedCall"^conv2d_22/StatefulPartitionedCall"^conv2d_23/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2F
!conv2d_21/StatefulPartitionedCall!conv2d_21/StatefulPartitionedCall2F
!conv2d_22/StatefulPartitionedCall!conv2d_22/StatefulPartitionedCall2F
!conv2d_23/StatefulPartitionedCall!conv2d_23/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
c
G__inference_flatten_3_layer_call_and_return_conditional_losses_44799797

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
?
?
$__inference__traced_restore_44800382
file_prefix;
!assignvariableop_conv2d_21_kernel: /
!assignvariableop_1_conv2d_21_bias: =
#assignvariableop_2_conv2d_22_kernel:@@/
!assignvariableop_3_conv2d_22_bias:@?
#assignvariableop_4_conv2d_23_kernel:??0
!assignvariableop_5_conv2d_23_bias:	?

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
AssignVariableOpAssignVariableOp!assignvariableop_conv2d_21_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp!assignvariableop_1_conv2d_21_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp#assignvariableop_2_conv2d_22_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp!assignvariableop_3_conv2d_22_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp#assignvariableop_4_conv2d_23_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp!assignvariableop_5_conv2d_23_biasIdentity_5:output:0"/device:CPU:0*
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
?3
?
F__inference_critic_3_layer_call_and_return_conditional_losses_44800042

inputsB
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource:@@7
)conv2d_22_biasadd_readvariableop_resource:@D
(conv2d_23_conv2d_readvariableop_resource:??8
)conv2d_23_biasadd_readvariableop_resource:	?
identity?? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOp?
conv2d_21/Conv2DConv2Dinputs'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_21/Conv2D?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_21/BiasAdd?
conv2d_21/CRelu/NegNegconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_21/CRelu/Negw
conv2d_21/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_21/CRelu/axis?
conv2d_21/CReluConcatV2conv2d_21/BiasAdd:output:0conv2d_21/CRelu/Neg:y:0conv2d_21/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_21/CRelu?
conv2d_21/CRelu/ReluReluconv2d_21/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_21/CRelu/Relu?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2D"conv2d_21/CRelu/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_22/BiasAdd?
conv2d_22/CRelu/NegNegconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_22/CRelu/Negw
conv2d_22/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_22/CRelu/axis?
conv2d_22/CReluConcatV2conv2d_22/BiasAdd:output:0conv2d_22/CRelu/Neg:y:0conv2d_22/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_22/CRelu?
conv2d_22/CRelu/ReluReluconv2d_22/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_22/CRelu/Relu?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2D"conv2d_22/CRelu/Relu:activations:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_23/BiasAdd?
conv2d_23/CRelu/NegNegconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu/Negw
conv2d_23/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_23/CRelu/axis?
conv2d_23/CReluConcatV2conv2d_23/BiasAdd:output:0conv2d_23/CRelu/Neg:y:0conv2d_23/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu?
conv2d_23/CRelu/ReluReluconv2d_23/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_3/Const?
flatten_3/ReshapeReshape"conv2d_23/CRelu/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_3/Reshape?
norm/mulMulflatten_3/Reshape:output:0flatten_3/Reshape:output:0*
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
truedivRealDivflatten_3/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?<
?
#__inference__wrapped_model_44799724
input_1K
1critic_3_conv2d_21_conv2d_readvariableop_resource: @
2critic_3_conv2d_21_biasadd_readvariableop_resource: K
1critic_3_conv2d_22_conv2d_readvariableop_resource:@@@
2critic_3_conv2d_22_biasadd_readvariableop_resource:@M
1critic_3_conv2d_23_conv2d_readvariableop_resource:??A
2critic_3_conv2d_23_biasadd_readvariableop_resource:	?
identity??)critic_3/conv2d_21/BiasAdd/ReadVariableOp?(critic_3/conv2d_21/Conv2D/ReadVariableOp?)critic_3/conv2d_22/BiasAdd/ReadVariableOp?(critic_3/conv2d_22/Conv2D/ReadVariableOp?)critic_3/conv2d_23/BiasAdd/ReadVariableOp?(critic_3/conv2d_23/Conv2D/ReadVariableOp?
(critic_3/conv2d_21/Conv2D/ReadVariableOpReadVariableOp1critic_3_conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02*
(critic_3/conv2d_21/Conv2D/ReadVariableOp?
critic_3/conv2d_21/Conv2DConv2Dinput_10critic_3/conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
critic_3/conv2d_21/Conv2D?
)critic_3/conv2d_21/BiasAdd/ReadVariableOpReadVariableOp2critic_3_conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)critic_3/conv2d_21/BiasAdd/ReadVariableOp?
critic_3/conv2d_21/BiasAddBiasAdd"critic_3/conv2d_21/Conv2D:output:01critic_3/conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
critic_3/conv2d_21/BiasAdd?
critic_3/conv2d_21/CRelu/NegNeg#critic_3/conv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
critic_3/conv2d_21/CRelu/Neg?
critic_3/conv2d_21/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
critic_3/conv2d_21/CRelu/axis?
critic_3/conv2d_21/CReluConcatV2#critic_3/conv2d_21/BiasAdd:output:0 critic_3/conv2d_21/CRelu/Neg:y:0&critic_3/conv2d_21/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
critic_3/conv2d_21/CRelu?
critic_3/conv2d_21/CRelu/ReluRelu!critic_3/conv2d_21/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
critic_3/conv2d_21/CRelu/Relu?
(critic_3/conv2d_22/Conv2D/ReadVariableOpReadVariableOp1critic_3_conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02*
(critic_3/conv2d_22/Conv2D/ReadVariableOp?
critic_3/conv2d_22/Conv2DConv2D+critic_3/conv2d_21/CRelu/Relu:activations:00critic_3/conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
critic_3/conv2d_22/Conv2D?
)critic_3/conv2d_22/BiasAdd/ReadVariableOpReadVariableOp2critic_3_conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02+
)critic_3/conv2d_22/BiasAdd/ReadVariableOp?
critic_3/conv2d_22/BiasAddBiasAdd"critic_3/conv2d_22/Conv2D:output:01critic_3/conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
critic_3/conv2d_22/BiasAdd?
critic_3/conv2d_22/CRelu/NegNeg#critic_3/conv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
critic_3/conv2d_22/CRelu/Neg?
critic_3/conv2d_22/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
critic_3/conv2d_22/CRelu/axis?
critic_3/conv2d_22/CReluConcatV2#critic_3/conv2d_22/BiasAdd:output:0 critic_3/conv2d_22/CRelu/Neg:y:0&critic_3/conv2d_22/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
critic_3/conv2d_22/CRelu?
critic_3/conv2d_22/CRelu/ReluRelu!critic_3/conv2d_22/CRelu:output:0*
T0*0
_output_shapes
:??????????2
critic_3/conv2d_22/CRelu/Relu?
(critic_3/conv2d_23/Conv2D/ReadVariableOpReadVariableOp1critic_3_conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02*
(critic_3/conv2d_23/Conv2D/ReadVariableOp?
critic_3/conv2d_23/Conv2DConv2D+critic_3/conv2d_22/CRelu/Relu:activations:00critic_3/conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
critic_3/conv2d_23/Conv2D?
)critic_3/conv2d_23/BiasAdd/ReadVariableOpReadVariableOp2critic_3_conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)critic_3/conv2d_23/BiasAdd/ReadVariableOp?
critic_3/conv2d_23/BiasAddBiasAdd"critic_3/conv2d_23/Conv2D:output:01critic_3/conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
critic_3/conv2d_23/BiasAdd?
critic_3/conv2d_23/CRelu/NegNeg#critic_3/conv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
critic_3/conv2d_23/CRelu/Neg?
critic_3/conv2d_23/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
critic_3/conv2d_23/CRelu/axis?
critic_3/conv2d_23/CReluConcatV2#critic_3/conv2d_23/BiasAdd:output:0 critic_3/conv2d_23/CRelu/Neg:y:0&critic_3/conv2d_23/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
critic_3/conv2d_23/CRelu?
critic_3/conv2d_23/CRelu/ReluRelu!critic_3/conv2d_23/CRelu:output:0*
T0*0
_output_shapes
:??????????2
critic_3/conv2d_23/CRelu/Relu?
critic_3/flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
critic_3/flatten_3/Const?
critic_3/flatten_3/ReshapeReshape+critic_3/conv2d_23/CRelu/Relu:activations:0!critic_3/flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????b2
critic_3/flatten_3/Reshape?
critic_3/norm/mulMul#critic_3/flatten_3/Reshape:output:0#critic_3/flatten_3/Reshape:output:0*
T0*(
_output_shapes
:??????????b2
critic_3/norm/mul?
#critic_3/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2%
#critic_3/norm/Sum/reduction_indices?
critic_3/norm/SumSumcritic_3/norm/mul:z:0,critic_3/norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
critic_3/norm/Sum~
critic_3/norm/SqrtSqrtcritic_3/norm/Sum:output:0*
T0*'
_output_shapes
:?????????2
critic_3/norm/Sqrt?
critic_3/truedivRealDiv#critic_3/flatten_3/Reshape:output:0critic_3/norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2
critic_3/truediv?
IdentityIdentitycritic_3/truediv:z:0*^critic_3/conv2d_21/BiasAdd/ReadVariableOp)^critic_3/conv2d_21/Conv2D/ReadVariableOp*^critic_3/conv2d_22/BiasAdd/ReadVariableOp)^critic_3/conv2d_22/Conv2D/ReadVariableOp*^critic_3/conv2d_23/BiasAdd/ReadVariableOp)^critic_3/conv2d_23/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2V
)critic_3/conv2d_21/BiasAdd/ReadVariableOp)critic_3/conv2d_21/BiasAdd/ReadVariableOp2T
(critic_3/conv2d_21/Conv2D/ReadVariableOp(critic_3/conv2d_21/Conv2D/ReadVariableOp2V
)critic_3/conv2d_22/BiasAdd/ReadVariableOp)critic_3/conv2d_22/BiasAdd/ReadVariableOp2T
(critic_3/conv2d_22/Conv2D/ReadVariableOp(critic_3/conv2d_22/Conv2D/ReadVariableOp2V
)critic_3/conv2d_23/BiasAdd/ReadVariableOp)critic_3/conv2d_23/BiasAdd/ReadVariableOp2T
(critic_3/conv2d_23/Conv2D/ReadVariableOp(critic_3/conv2d_23/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
,__inference_conv2d_22_layer_call_fn_44800279

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
GPU2*0,1J 8? *P
fKRI
G__inference_conv2d_22_layer_call_and_return_conditional_losses_447997652
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
?3
?
F__inference_critic_3_layer_call_and_return_conditional_losses_44800083

inputsB
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource:@@7
)conv2d_22_biasadd_readvariableop_resource:@D
(conv2d_23_conv2d_readvariableop_resource:??8
)conv2d_23_biasadd_readvariableop_resource:	?
identity?? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOp?
conv2d_21/Conv2DConv2Dinputs'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_21/Conv2D?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_21/BiasAdd?
conv2d_21/CRelu/NegNegconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_21/CRelu/Negw
conv2d_21/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_21/CRelu/axis?
conv2d_21/CReluConcatV2conv2d_21/BiasAdd:output:0conv2d_21/CRelu/Neg:y:0conv2d_21/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_21/CRelu?
conv2d_21/CRelu/ReluReluconv2d_21/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_21/CRelu/Relu?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2D"conv2d_21/CRelu/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_22/BiasAdd?
conv2d_22/CRelu/NegNegconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_22/CRelu/Negw
conv2d_22/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_22/CRelu/axis?
conv2d_22/CReluConcatV2conv2d_22/BiasAdd:output:0conv2d_22/CRelu/Neg:y:0conv2d_22/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_22/CRelu?
conv2d_22/CRelu/ReluReluconv2d_22/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_22/CRelu/Relu?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2D"conv2d_22/CRelu/Relu:activations:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_23/BiasAdd?
conv2d_23/CRelu/NegNegconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu/Negw
conv2d_23/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_23/CRelu/axis?
conv2d_23/CReluConcatV2conv2d_23/BiasAdd:output:0conv2d_23/CRelu/Neg:y:0conv2d_23/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu?
conv2d_23/CRelu/ReluReluconv2d_23/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_3/Const?
flatten_3/ReshapeReshape"conv2d_23/CRelu/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_3/Reshape?
norm/mulMulflatten_3/Reshape:output:0flatten_3/Reshape:output:0*
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
truedivRealDivflatten_3/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
G__inference_conv2d_22_layer_call_and_return_conditional_losses_44799765

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
F__inference_critic_3_layer_call_and_return_conditional_losses_44800124
input_1B
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource:@@7
)conv2d_22_biasadd_readvariableop_resource:@D
(conv2d_23_conv2d_readvariableop_resource:??8
)conv2d_23_biasadd_readvariableop_resource:	?
identity?? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOp?
conv2d_21/Conv2DConv2Dinput_1'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_21/Conv2D?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_21/BiasAdd?
conv2d_21/CRelu/NegNegconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_21/CRelu/Negw
conv2d_21/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_21/CRelu/axis?
conv2d_21/CReluConcatV2conv2d_21/BiasAdd:output:0conv2d_21/CRelu/Neg:y:0conv2d_21/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_21/CRelu?
conv2d_21/CRelu/ReluReluconv2d_21/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_21/CRelu/Relu?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2D"conv2d_21/CRelu/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_22/BiasAdd?
conv2d_22/CRelu/NegNegconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_22/CRelu/Negw
conv2d_22/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_22/CRelu/axis?
conv2d_22/CReluConcatV2conv2d_22/BiasAdd:output:0conv2d_22/CRelu/Neg:y:0conv2d_22/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_22/CRelu?
conv2d_22/CRelu/ReluReluconv2d_22/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_22/CRelu/Relu?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2D"conv2d_22/CRelu/Relu:activations:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_23/BiasAdd?
conv2d_23/CRelu/NegNegconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu/Negw
conv2d_23/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_23/CRelu/axis?
conv2d_23/CReluConcatV2conv2d_23/BiasAdd:output:0conv2d_23/CRelu/Neg:y:0conv2d_23/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu?
conv2d_23/CRelu/ReluReluconv2d_23/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_3/Const?
flatten_3/ReshapeReshape"conv2d_23/CRelu/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_3/Reshape?
norm/mulMulflatten_3/Reshape:output:0flatten_3/Reshape:output:0*
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
truedivRealDivflatten_3/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
G__inference_conv2d_23_layer_call_and_return_conditional_losses_44799785

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
G__inference_conv2d_22_layer_call_and_return_conditional_losses_44800270

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
?
?
,__inference_conv2d_23_layer_call_fn_44800302

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
GPU2*0,1J 8? *P
fKRI
G__inference_conv2d_23_layer_call_and_return_conditional_losses_447997852
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
?3
?
F__inference_critic_3_layer_call_and_return_conditional_losses_44800165
input_1B
(conv2d_21_conv2d_readvariableop_resource: 7
)conv2d_21_biasadd_readvariableop_resource: B
(conv2d_22_conv2d_readvariableop_resource:@@7
)conv2d_22_biasadd_readvariableop_resource:@D
(conv2d_23_conv2d_readvariableop_resource:??8
)conv2d_23_biasadd_readvariableop_resource:	?
identity?? conv2d_21/BiasAdd/ReadVariableOp?conv2d_21/Conv2D/ReadVariableOp? conv2d_22/BiasAdd/ReadVariableOp?conv2d_22/Conv2D/ReadVariableOp? conv2d_23/BiasAdd/ReadVariableOp?conv2d_23/Conv2D/ReadVariableOp?
conv2d_21/Conv2D/ReadVariableOpReadVariableOp(conv2d_21_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_21/Conv2D/ReadVariableOp?
conv2d_21/Conv2DConv2Dinput_1'conv2d_21/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_21/Conv2D?
 conv2d_21/BiasAdd/ReadVariableOpReadVariableOp)conv2d_21_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02"
 conv2d_21/BiasAdd/ReadVariableOp?
conv2d_21/BiasAddBiasAddconv2d_21/Conv2D:output:0(conv2d_21/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_21/BiasAdd?
conv2d_21/CRelu/NegNegconv2d_21/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_21/CRelu/Negw
conv2d_21/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_21/CRelu/axis?
conv2d_21/CReluConcatV2conv2d_21/BiasAdd:output:0conv2d_21/CRelu/Neg:y:0conv2d_21/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_21/CRelu?
conv2d_21/CRelu/ReluReluconv2d_21/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_21/CRelu/Relu?
conv2d_22/Conv2D/ReadVariableOpReadVariableOp(conv2d_22_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02!
conv2d_22/Conv2D/ReadVariableOp?
conv2d_22/Conv2DConv2D"conv2d_21/CRelu/Relu:activations:0'conv2d_22/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_22/Conv2D?
 conv2d_22/BiasAdd/ReadVariableOpReadVariableOp)conv2d_22_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02"
 conv2d_22/BiasAdd/ReadVariableOp?
conv2d_22/BiasAddBiasAddconv2d_22/Conv2D:output:0(conv2d_22/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_22/BiasAdd?
conv2d_22/CRelu/NegNegconv2d_22/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_22/CRelu/Negw
conv2d_22/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_22/CRelu/axis?
conv2d_22/CReluConcatV2conv2d_22/BiasAdd:output:0conv2d_22/CRelu/Neg:y:0conv2d_22/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_22/CRelu?
conv2d_22/CRelu/ReluReluconv2d_22/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_22/CRelu/Relu?
conv2d_23/Conv2D/ReadVariableOpReadVariableOp(conv2d_23_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_23/Conv2D/ReadVariableOp?
conv2d_23/Conv2DConv2D"conv2d_22/CRelu/Relu:activations:0'conv2d_23/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_23/Conv2D?
 conv2d_23/BiasAdd/ReadVariableOpReadVariableOp)conv2d_23_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02"
 conv2d_23/BiasAdd/ReadVariableOp?
conv2d_23/BiasAddBiasAddconv2d_23/Conv2D:output:0(conv2d_23/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_23/BiasAdd?
conv2d_23/CRelu/NegNegconv2d_23/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu/Negw
conv2d_23/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_23/CRelu/axis?
conv2d_23/CReluConcatV2conv2d_23/BiasAdd:output:0conv2d_23/CRelu/Neg:y:0conv2d_23/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu?
conv2d_23/CRelu/ReluReluconv2d_23/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_23/CRelu/Relus
flatten_3/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten_3/Const?
flatten_3/ReshapeReshape"conv2d_23/CRelu/Relu:activations:0flatten_3/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten_3/Reshape?
norm/mulMulflatten_3/Reshape:output:0flatten_3/Reshape:output:0*
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
truedivRealDivflatten_3/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_21/BiasAdd/ReadVariableOp ^conv2d_21/Conv2D/ReadVariableOp!^conv2d_22/BiasAdd/ReadVariableOp ^conv2d_22/Conv2D/ReadVariableOp!^conv2d_23/BiasAdd/ReadVariableOp ^conv2d_23/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_21/BiasAdd/ReadVariableOp conv2d_21/BiasAdd/ReadVariableOp2B
conv2d_21/Conv2D/ReadVariableOpconv2d_21/Conv2D/ReadVariableOp2D
 conv2d_22/BiasAdd/ReadVariableOp conv2d_22/BiasAdd/ReadVariableOp2B
conv2d_22/Conv2D/ReadVariableOpconv2d_22/Conv2D/ReadVariableOp2D
 conv2d_23/BiasAdd/ReadVariableOp conv2d_23/BiasAdd/ReadVariableOp2B
conv2d_23/Conv2D/ReadVariableOpconv2d_23/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
H
,__inference_flatten_3_layer_call_fn_44800313

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
GPU2*0,1J 8? *P
fKRI
G__inference_flatten_3_layer_call_and_return_conditional_losses_447997972
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
 
_user_specified_nameinputs"?L
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
_tf_keras_model?{"name": "critic_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Critic", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 28, 28, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Critic"}}
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
_tf_keras_layer?	{"name": "conv2d_21", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_21", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "crelu_v2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*>&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_22", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_22", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "crelu_v2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 64]}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"?	
_tf_keras_layer?	{"name": "conv2d_23", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_23", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "crelu_v2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 128]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*B&call_and_return_all_conditional_losses
C__call__"?
_tf_keras_layer?{"name": "flatten_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten_3", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 13}}
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
*:( 2conv2d_21/kernel
: 2conv2d_21/bias
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
*:(@@2conv2d_22/kernel
:@2conv2d_22/bias
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
,:*??2conv2d_23/kernel
:?2conv2d_23/bias
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
F__inference_critic_3_layer_call_and_return_conditional_losses_44800042
F__inference_critic_3_layer_call_and_return_conditional_losses_44800083
F__inference_critic_3_layer_call_and_return_conditional_losses_44800124
F__inference_critic_3_layer_call_and_return_conditional_losses_44800165?
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
+__inference_critic_3_layer_call_fn_44800182
+__inference_critic_3_layer_call_fn_44800199
+__inference_critic_3_layer_call_fn_44800216
+__inference_critic_3_layer_call_fn_44800233?
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
#__inference__wrapped_model_44799724?
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
G__inference_conv2d_21_layer_call_and_return_conditional_losses_44800247?
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
,__inference_conv2d_21_layer_call_fn_44800256?
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
G__inference_conv2d_22_layer_call_and_return_conditional_losses_44800270?
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
,__inference_conv2d_22_layer_call_fn_44800279?
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
G__inference_conv2d_23_layer_call_and_return_conditional_losses_44800293?
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
,__inference_conv2d_23_layer_call_fn_44800302?
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
G__inference_flatten_3_layer_call_and_return_conditional_losses_44800308?
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
,__inference_flatten_3_layer_call_fn_44800313?
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
&__inference_signature_wrapper_44800001input_1"?
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
#__inference__wrapped_model_44799724x
8?5
.?+
)?&
input_1?????????
? "4?1
/
output_1#? 
output_1??????????b?
G__inference_conv2d_21_layer_call_and_return_conditional_losses_44800247l
7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????@
? ?
,__inference_conv2d_21_layer_call_fn_44800256_
7?4
-?*
(?%
inputs?????????
? " ??????????@?
G__inference_conv2d_22_layer_call_and_return_conditional_losses_44800270m7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_22_layer_call_fn_44800279`7?4
-?*
(?%
inputs?????????@
? "!????????????
G__inference_conv2d_23_layer_call_and_return_conditional_losses_44800293n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
,__inference_conv2d_23_layer_call_fn_44800302a8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_critic_3_layer_call_and_return_conditional_losses_44800042m
;?8
1?.
(?%
inputs?????????
p 
? "&?#
?
0??????????b
? ?
F__inference_critic_3_layer_call_and_return_conditional_losses_44800083m
;?8
1?.
(?%
inputs?????????
p
? "&?#
?
0??????????b
? ?
F__inference_critic_3_layer_call_and_return_conditional_losses_44800124n
<?9
2?/
)?&
input_1?????????
p 
? "&?#
?
0??????????b
? ?
F__inference_critic_3_layer_call_and_return_conditional_losses_44800165n
<?9
2?/
)?&
input_1?????????
p
? "&?#
?
0??????????b
? ?
+__inference_critic_3_layer_call_fn_44800182a
<?9
2?/
)?&
input_1?????????
p 
? "???????????b?
+__inference_critic_3_layer_call_fn_44800199`
;?8
1?.
(?%
inputs?????????
p 
? "???????????b?
+__inference_critic_3_layer_call_fn_44800216`
;?8
1?.
(?%
inputs?????????
p
? "???????????b?
+__inference_critic_3_layer_call_fn_44800233a
<?9
2?/
)?&
input_1?????????
p
? "???????????b?
G__inference_flatten_3_layer_call_and_return_conditional_losses_44800308b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????b
? ?
,__inference_flatten_3_layer_call_fn_44800313U8?5
.?+
)?&
inputs??????????
? "???????????b?
&__inference_signature_wrapper_44800001?
C?@
? 
9?6
4
input_1)?&
input_1?????????"4?1
/
output_1#? 
output_1??????????b