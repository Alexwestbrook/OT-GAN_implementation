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
conv2d_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: * 
shared_nameconv2d_3/kernel
{
#conv2d_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_3/kernel*&
_output_shapes
: *
dtype0
r
conv2d_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameconv2d_3/bias
k
!conv2d_3/bias/Read/ReadVariableOpReadVariableOpconv2d_3/bias*
_output_shapes
: *
dtype0
?
conv2d_4/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@* 
shared_nameconv2d_4/kernel
{
#conv2d_4/kernel/Read/ReadVariableOpReadVariableOpconv2d_4/kernel*&
_output_shapes
:@@*
dtype0
r
conv2d_4/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_nameconv2d_4/bias
k
!conv2d_4/bias/Read/ReadVariableOpReadVariableOpconv2d_4/bias*
_output_shapes
:@*
dtype0
?
conv2d_5/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_5/kernel
}
#conv2d_5/kernel/Read/ReadVariableOpReadVariableOpconv2d_5/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_5/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_5/bias
l
!conv2d_5/bias/Read/ReadVariableOpReadVariableOpconv2d_5/bias*
_output_shapes	
:?*
dtype0

NoOpNoOp
?
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?
value?B? B?
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
trainable_variables

 layers
!layer_metrics
"layer_regularization_losses
#metrics
regularization_losses
$non_trainable_variables
	variables
 
LJ
VARIABLE_VALUEconv2d_3/kernel'conv1/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_3/bias%conv1/bias/.ATTRIBUTES/VARIABLE_VALUE
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
&layer_metrics
trainable_variables
'layer_regularization_losses
(metrics
regularization_losses
)non_trainable_variables
	variables
LJ
VARIABLE_VALUEconv2d_4/kernel'conv2/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_4/bias%conv2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

*layers
+layer_metrics
trainable_variables
,layer_regularization_losses
-metrics
regularization_losses
.non_trainable_variables
	variables
LJ
VARIABLE_VALUEconv2d_5/kernel'conv3/kernel/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUEconv2d_5/bias%conv3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
?

/layers
0layer_metrics
trainable_variables
1layer_regularization_losses
2metrics
regularization_losses
3non_trainable_variables
	variables
 
 
 
?

4layers
5layer_metrics
trainable_variables
6layer_regularization_losses
7metrics
regularization_losses
8non_trainable_variables
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
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/bias*
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
&__inference_signature_wrapper_13833109
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename#conv2d_3/kernel/Read/ReadVariableOp!conv2d_3/bias/Read/ReadVariableOp#conv2d_4/kernel/Read/ReadVariableOp!conv2d_4/bias/Read/ReadVariableOp#conv2d_5/kernel/Read/ReadVariableOp!conv2d_5/bias/Read/ReadVariableOpConst*
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
!__inference__traced_save_13833462
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d_3/kernelconv2d_3/biasconv2d_4/kernelconv2d_4/biasconv2d_5/kernelconv2d_5/bias*
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
$__inference__traced_restore_13833490??
?
?
+__inference_conv2d_5_layer_call_fn_13833410

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
GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_138328932
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
?
?
$__inference__traced_restore_13833490
file_prefix:
 assignvariableop_conv2d_3_kernel: .
 assignvariableop_1_conv2d_3_bias: <
"assignvariableop_2_conv2d_4_kernel:@@.
 assignvariableop_3_conv2d_4_bias:@>
"assignvariableop_4_conv2d_5_kernel:??/
 assignvariableop_5_conv2d_5_bias:	?

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
AssignVariableOpAssignVariableOp assignvariableop_conv2d_3_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp assignvariableop_1_conv2d_3_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp"assignvariableop_2_conv2d_4_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp assignvariableop_3_conv2d_4_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp"assignvariableop_4_conv2d_5_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp assignvariableop_5_conv2d_5_biasIdentity_5:output:0"/device:CPU:0*
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
?2
?
D__inference_critic_layer_call_and_return_conditional_losses_13833232
input_1A
'conv2d_3_conv2d_readvariableop_resource: 6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@C
'conv2d_5_conv2d_readvariableop_resource:??7
(conv2d_5_biasadd_readvariableop_resource:	?
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinput_1&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/BiasAdd?
conv2d_3/CRelu/NegNegconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/CRelu/Negu
conv2d_3/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_3/CRelu/axis?
conv2d_3/CReluConcatV2conv2d_3/BiasAdd:output:0conv2d_3/CRelu/Neg:y:0conv2d_3/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_3/CRelu?
conv2d_3/CRelu/ReluReluconv2d_3/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/CRelu/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D!conv2d_3/CRelu/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/BiasAdd?
conv2d_4/CRelu/NegNegconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/CRelu/Negu
conv2d_4/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_4/CRelu/axis?
conv2d_4/CReluConcatV2conv2d_4/BiasAdd:output:0conv2d_4/CRelu/Neg:y:0conv2d_4/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_4/CRelu?
conv2d_4/CRelu/ReluReluconv2d_4/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/CRelu/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D!conv2d_4/CRelu/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
conv2d_5/CRelu/NegNegconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu/Negu
conv2d_5/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_5/CRelu/axis?
conv2d_5/CReluConcatV2conv2d_5/BiasAdd:output:0conv2d_5/CRelu/Neg:y:0conv2d_5/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu?
conv2d_5/CRelu/ReluReluconv2d_5/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten/Const?
flatten/ReshapeReshape!conv2d_5/CRelu/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten/Reshape?
norm/mulMulflatten/Reshape:output:0flatten/Reshape:output:0*
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
	norm/Sqrty
truedivRealDivflatten/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
a
E__inference_flatten_layer_call_and_return_conditional_losses_13833416

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
!__inference__traced_save_13833462
file_prefix.
*savev2_conv2d_3_kernel_read_readvariableop,
(savev2_conv2d_3_bias_read_readvariableop.
*savev2_conv2d_4_kernel_read_readvariableop,
(savev2_conv2d_4_bias_read_readvariableop.
*savev2_conv2d_5_kernel_read_readvariableop,
(savev2_conv2d_5_bias_read_readvariableop
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
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0*savev2_conv2d_3_kernel_read_readvariableop(savev2_conv2d_3_bias_read_readvariableop*savev2_conv2d_4_kernel_read_readvariableop(savev2_conv2d_4_bias_read_readvariableop*savev2_conv2d_5_kernel_read_readvariableop(savev2_conv2d_5_bias_read_readvariableopsavev2_const"/device:CPU:0*
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
?
?
F__inference_conv2d_5_layer_call_and_return_conditional_losses_13833401

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
)__inference_critic_layer_call_fn_13833290
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
GPU2*0,1J 8? *M
fHRF
D__inference_critic_layer_call_and_return_conditional_losses_138329132
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
a
E__inference_flatten_layer_call_and_return_conditional_losses_13832905

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
?9
?
#__inference__wrapped_model_13832832
input_1H
.critic_conv2d_3_conv2d_readvariableop_resource: =
/critic_conv2d_3_biasadd_readvariableop_resource: H
.critic_conv2d_4_conv2d_readvariableop_resource:@@=
/critic_conv2d_4_biasadd_readvariableop_resource:@J
.critic_conv2d_5_conv2d_readvariableop_resource:??>
/critic_conv2d_5_biasadd_readvariableop_resource:	?
identity??&critic/conv2d_3/BiasAdd/ReadVariableOp?%critic/conv2d_3/Conv2D/ReadVariableOp?&critic/conv2d_4/BiasAdd/ReadVariableOp?%critic/conv2d_4/Conv2D/ReadVariableOp?&critic/conv2d_5/BiasAdd/ReadVariableOp?%critic/conv2d_5/Conv2D/ReadVariableOp?
%critic/conv2d_3/Conv2D/ReadVariableOpReadVariableOp.critic_conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02'
%critic/conv2d_3/Conv2D/ReadVariableOp?
critic/conv2d_3/Conv2DConv2Dinput_1-critic/conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
critic/conv2d_3/Conv2D?
&critic/conv2d_3/BiasAdd/ReadVariableOpReadVariableOp/critic_conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02(
&critic/conv2d_3/BiasAdd/ReadVariableOp?
critic/conv2d_3/BiasAddBiasAddcritic/conv2d_3/Conv2D:output:0.critic/conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
critic/conv2d_3/BiasAdd?
critic/conv2d_3/CRelu/NegNeg critic/conv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
critic/conv2d_3/CRelu/Neg?
critic/conv2d_3/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
critic/conv2d_3/CRelu/axis?
critic/conv2d_3/CReluConcatV2 critic/conv2d_3/BiasAdd:output:0critic/conv2d_3/CRelu/Neg:y:0#critic/conv2d_3/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
critic/conv2d_3/CRelu?
critic/conv2d_3/CRelu/ReluRelucritic/conv2d_3/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
critic/conv2d_3/CRelu/Relu?
%critic/conv2d_4/Conv2D/ReadVariableOpReadVariableOp.critic_conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02'
%critic/conv2d_4/Conv2D/ReadVariableOp?
critic/conv2d_4/Conv2DConv2D(critic/conv2d_3/CRelu/Relu:activations:0-critic/conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
critic/conv2d_4/Conv2D?
&critic/conv2d_4/BiasAdd/ReadVariableOpReadVariableOp/critic_conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02(
&critic/conv2d_4/BiasAdd/ReadVariableOp?
critic/conv2d_4/BiasAddBiasAddcritic/conv2d_4/Conv2D:output:0.critic/conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
critic/conv2d_4/BiasAdd?
critic/conv2d_4/CRelu/NegNeg critic/conv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
critic/conv2d_4/CRelu/Neg?
critic/conv2d_4/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
critic/conv2d_4/CRelu/axis?
critic/conv2d_4/CReluConcatV2 critic/conv2d_4/BiasAdd:output:0critic/conv2d_4/CRelu/Neg:y:0#critic/conv2d_4/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
critic/conv2d_4/CRelu?
critic/conv2d_4/CRelu/ReluRelucritic/conv2d_4/CRelu:output:0*
T0*0
_output_shapes
:??????????2
critic/conv2d_4/CRelu/Relu?
%critic/conv2d_5/Conv2D/ReadVariableOpReadVariableOp.critic_conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02'
%critic/conv2d_5/Conv2D/ReadVariableOp?
critic/conv2d_5/Conv2DConv2D(critic/conv2d_4/CRelu/Relu:activations:0-critic/conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
critic/conv2d_5/Conv2D?
&critic/conv2d_5/BiasAdd/ReadVariableOpReadVariableOp/critic_conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&critic/conv2d_5/BiasAdd/ReadVariableOp?
critic/conv2d_5/BiasAddBiasAddcritic/conv2d_5/Conv2D:output:0.critic/conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
critic/conv2d_5/BiasAdd?
critic/conv2d_5/CRelu/NegNeg critic/conv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
critic/conv2d_5/CRelu/Neg?
critic/conv2d_5/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
critic/conv2d_5/CRelu/axis?
critic/conv2d_5/CReluConcatV2 critic/conv2d_5/BiasAdd:output:0critic/conv2d_5/CRelu/Neg:y:0#critic/conv2d_5/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
critic/conv2d_5/CRelu?
critic/conv2d_5/CRelu/ReluRelucritic/conv2d_5/CRelu:output:0*
T0*0
_output_shapes
:??????????2
critic/conv2d_5/CRelu/Relu}
critic/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
critic/flatten/Const?
critic/flatten/ReshapeReshape(critic/conv2d_5/CRelu/Relu:activations:0critic/flatten/Const:output:0*
T0*(
_output_shapes
:??????????b2
critic/flatten/Reshape?
critic/norm/mulMulcritic/flatten/Reshape:output:0critic/flatten/Reshape:output:0*
T0*(
_output_shapes
:??????????b2
critic/norm/mul?
!critic/norm/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*
valueB:2#
!critic/norm/Sum/reduction_indices?
critic/norm/SumSumcritic/norm/mul:z:0*critic/norm/Sum/reduction_indices:output:0*
T0*'
_output_shapes
:?????????*
	keep_dims(2
critic/norm/Sumx
critic/norm/SqrtSqrtcritic/norm/Sum:output:0*
T0*'
_output_shapes
:?????????2
critic/norm/Sqrt?
critic/truedivRealDivcritic/flatten/Reshape:output:0critic/norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2
critic/truediv?
IdentityIdentitycritic/truediv:z:0'^critic/conv2d_3/BiasAdd/ReadVariableOp&^critic/conv2d_3/Conv2D/ReadVariableOp'^critic/conv2d_4/BiasAdd/ReadVariableOp&^critic/conv2d_4/Conv2D/ReadVariableOp'^critic/conv2d_5/BiasAdd/ReadVariableOp&^critic/conv2d_5/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2P
&critic/conv2d_3/BiasAdd/ReadVariableOp&critic/conv2d_3/BiasAdd/ReadVariableOp2N
%critic/conv2d_3/Conv2D/ReadVariableOp%critic/conv2d_3/Conv2D/ReadVariableOp2P
&critic/conv2d_4/BiasAdd/ReadVariableOp&critic/conv2d_4/BiasAdd/ReadVariableOp2N
%critic/conv2d_4/Conv2D/ReadVariableOp%critic/conv2d_4/Conv2D/ReadVariableOp2P
&critic/conv2d_5/BiasAdd/ReadVariableOp&critic/conv2d_5/BiasAdd/ReadVariableOp2N
%critic/conv2d_5/Conv2D/ReadVariableOp%critic/conv2d_5/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
?
F__inference_conv2d_4_layer_call_and_return_conditional_losses_13833378

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
?
?
)__inference_critic_layer_call_fn_13833324

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
GPU2*0,1J 8? *M
fHRF
D__inference_critic_layer_call_and_return_conditional_losses_138330082
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
?2
?
D__inference_critic_layer_call_and_return_conditional_losses_13833150

inputsA
'conv2d_3_conv2d_readvariableop_resource: 6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@C
'conv2d_5_conv2d_readvariableop_resource:??7
(conv2d_5_biasadd_readvariableop_resource:	?
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/BiasAdd?
conv2d_3/CRelu/NegNegconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/CRelu/Negu
conv2d_3/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_3/CRelu/axis?
conv2d_3/CReluConcatV2conv2d_3/BiasAdd:output:0conv2d_3/CRelu/Neg:y:0conv2d_3/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_3/CRelu?
conv2d_3/CRelu/ReluReluconv2d_3/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/CRelu/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D!conv2d_3/CRelu/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/BiasAdd?
conv2d_4/CRelu/NegNegconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/CRelu/Negu
conv2d_4/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_4/CRelu/axis?
conv2d_4/CReluConcatV2conv2d_4/BiasAdd:output:0conv2d_4/CRelu/Neg:y:0conv2d_4/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_4/CRelu?
conv2d_4/CRelu/ReluReluconv2d_4/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/CRelu/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D!conv2d_4/CRelu/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
conv2d_5/CRelu/NegNegconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu/Negu
conv2d_5/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_5/CRelu/axis?
conv2d_5/CReluConcatV2conv2d_5/BiasAdd:output:0conv2d_5/CRelu/Neg:y:0conv2d_5/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu?
conv2d_5/CRelu/ReluReluconv2d_5/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten/Const?
flatten/ReshapeReshape!conv2d_5/CRelu/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten/Reshape?
norm/mulMulflatten/Reshape:output:0flatten/Reshape:output:0*
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
	norm/Sqrty
truedivRealDivflatten/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
D__inference_critic_layer_call_and_return_conditional_losses_13832913

inputs+
conv2d_3_13832854: 
conv2d_3_13832856: +
conv2d_4_13832874:@@
conv2d_4_13832876:@-
conv2d_5_13832894:?? 
conv2d_5_13832896:	?
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_13832854conv2d_3_13832856*
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
GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_138328532"
 conv2d_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_13832874conv2d_4_13832876*
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
GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_138328732"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_13832894conv2d_5_13832896*
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
GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_138328932"
 conv2d_5/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_138329052
flatten/PartitionedCall?
norm/mulMul flatten/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
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
truedivRealDiv flatten/PartitionedCall:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
)__inference_critic_layer_call_fn_13833341
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
GPU2*0,1J 8? *M
fHRF
D__inference_critic_layer_call_and_return_conditional_losses_138330082
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
D__inference_critic_layer_call_and_return_conditional_losses_13833008

inputs+
conv2d_3_13832986: 
conv2d_3_13832988: +
conv2d_4_13832991:@@
conv2d_4_13832993:@-
conv2d_5_13832996:?? 
conv2d_5_13832998:	?
identity?? conv2d_3/StatefulPartitionedCall? conv2d_4/StatefulPartitionedCall? conv2d_5/StatefulPartitionedCall?
 conv2d_3/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_3_13832986conv2d_3_13832988*
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
GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_138328532"
 conv2d_3/StatefulPartitionedCall?
 conv2d_4/StatefulPartitionedCallStatefulPartitionedCall)conv2d_3/StatefulPartitionedCall:output:0conv2d_4_13832991conv2d_4_13832993*
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
GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_138328732"
 conv2d_4/StatefulPartitionedCall?
 conv2d_5/StatefulPartitionedCallStatefulPartitionedCall)conv2d_4/StatefulPartitionedCall:output:0conv2d_5_13832996conv2d_5_13832998*
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
GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_5_layer_call_and_return_conditional_losses_138328932"
 conv2d_5/StatefulPartitionedCall?
flatten/PartitionedCallPartitionedCall)conv2d_5/StatefulPartitionedCall:output:0*
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
GPU2*0,1J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_138329052
flatten/PartitionedCall?
norm/mulMul flatten/PartitionedCall:output:0 flatten/PartitionedCall:output:0*
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
truedivRealDiv flatten/PartitionedCall:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0!^conv2d_3/StatefulPartitionedCall!^conv2d_4/StatefulPartitionedCall!^conv2d_5/StatefulPartitionedCall*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2D
 conv2d_3/StatefulPartitionedCall conv2d_3/StatefulPartitionedCall2D
 conv2d_4/StatefulPartitionedCall conv2d_4/StatefulPartitionedCall2D
 conv2d_5/StatefulPartitionedCall conv2d_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?2
?
D__inference_critic_layer_call_and_return_conditional_losses_13833273
input_1A
'conv2d_3_conv2d_readvariableop_resource: 6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@C
'conv2d_5_conv2d_readvariableop_resource:??7
(conv2d_5_biasadd_readvariableop_resource:	?
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinput_1&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/BiasAdd?
conv2d_3/CRelu/NegNegconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/CRelu/Negu
conv2d_3/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_3/CRelu/axis?
conv2d_3/CReluConcatV2conv2d_3/BiasAdd:output:0conv2d_3/CRelu/Neg:y:0conv2d_3/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_3/CRelu?
conv2d_3/CRelu/ReluReluconv2d_3/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/CRelu/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D!conv2d_3/CRelu/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/BiasAdd?
conv2d_4/CRelu/NegNegconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/CRelu/Negu
conv2d_4/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_4/CRelu/axis?
conv2d_4/CReluConcatV2conv2d_4/BiasAdd:output:0conv2d_4/CRelu/Neg:y:0conv2d_4/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_4/CRelu?
conv2d_4/CRelu/ReluReluconv2d_4/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/CRelu/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D!conv2d_4/CRelu/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
conv2d_5/CRelu/NegNegconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu/Negu
conv2d_5/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_5/CRelu/axis?
conv2d_5/CReluConcatV2conv2d_5/BiasAdd:output:0conv2d_5/CRelu/Neg:y:0conv2d_5/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu?
conv2d_5/CRelu/ReluReluconv2d_5/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten/Const?
flatten/ReshapeReshape!conv2d_5/CRelu/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten/Reshape?
norm/mulMulflatten/Reshape:output:0flatten/Reshape:output:0*
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
	norm/Sqrty
truedivRealDivflatten/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?2
?
D__inference_critic_layer_call_and_return_conditional_losses_13833191

inputsA
'conv2d_3_conv2d_readvariableop_resource: 6
(conv2d_3_biasadd_readvariableop_resource: A
'conv2d_4_conv2d_readvariableop_resource:@@6
(conv2d_4_biasadd_readvariableop_resource:@C
'conv2d_5_conv2d_readvariableop_resource:??7
(conv2d_5_biasadd_readvariableop_resource:	?
identity??conv2d_3/BiasAdd/ReadVariableOp?conv2d_3/Conv2D/ReadVariableOp?conv2d_4/BiasAdd/ReadVariableOp?conv2d_4/Conv2D/ReadVariableOp?conv2d_5/BiasAdd/ReadVariableOp?conv2d_5/Conv2D/ReadVariableOp?
conv2d_3/Conv2D/ReadVariableOpReadVariableOp'conv2d_3_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02 
conv2d_3/Conv2D/ReadVariableOp?
conv2d_3/Conv2DConv2Dinputs&conv2d_3/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2
conv2d_3/Conv2D?
conv2d_3/BiasAdd/ReadVariableOpReadVariableOp(conv2d_3_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02!
conv2d_3/BiasAdd/ReadVariableOp?
conv2d_3/BiasAddBiasAddconv2d_3/Conv2D:output:0'conv2d_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/BiasAdd?
conv2d_3/CRelu/NegNegconv2d_3/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_3/CRelu/Negu
conv2d_3/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_3/CRelu/axis?
conv2d_3/CReluConcatV2conv2d_3/BiasAdd:output:0conv2d_3/CRelu/Neg:y:0conv2d_3/CRelu/axis:output:0*
N*
T0*/
_output_shapes
:?????????@2
conv2d_3/CRelu?
conv2d_3/CRelu/ReluReluconv2d_3/CRelu:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_3/CRelu/Relu?
conv2d_4/Conv2D/ReadVariableOpReadVariableOp'conv2d_4_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02 
conv2d_4/Conv2D/ReadVariableOp?
conv2d_4/Conv2DConv2D!conv2d_3/CRelu/Relu:activations:0&conv2d_4/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2d_4/Conv2D?
conv2d_4/BiasAdd/ReadVariableOpReadVariableOp(conv2d_4_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_4/BiasAdd/ReadVariableOp?
conv2d_4/BiasAddBiasAddconv2d_4/Conv2D:output:0'conv2d_4/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/BiasAdd?
conv2d_4/CRelu/NegNegconv2d_4/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_4/CRelu/Negu
conv2d_4/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_4/CRelu/axis?
conv2d_4/CReluConcatV2conv2d_4/BiasAdd:output:0conv2d_4/CRelu/Neg:y:0conv2d_4/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_4/CRelu?
conv2d_4/CRelu/ReluReluconv2d_4/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_4/CRelu/Relu?
conv2d_5/Conv2D/ReadVariableOpReadVariableOp'conv2d_5_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_5/Conv2D/ReadVariableOp?
conv2d_5/Conv2DConv2D!conv2d_4/CRelu/Relu:activations:0&conv2d_5/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingSAME*
strides
2
conv2d_5/Conv2D?
conv2d_5/BiasAdd/ReadVariableOpReadVariableOp(conv2d_5_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_5/BiasAdd/ReadVariableOp?
conv2d_5/BiasAddBiasAddconv2d_5/Conv2D:output:0'conv2d_5/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_5/BiasAdd?
conv2d_5/CRelu/NegNegconv2d_5/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu/Negu
conv2d_5/CRelu/axisConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
conv2d_5/CRelu/axis?
conv2d_5/CReluConcatV2conv2d_5/BiasAdd:output:0conv2d_5/CRelu/Neg:y:0conv2d_5/CRelu/axis:output:0*
N*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu?
conv2d_5/CRelu/ReluReluconv2d_5/CRelu:output:0*
T0*0
_output_shapes
:??????????2
conv2d_5/CRelu/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 1  2
flatten/Const?
flatten/ReshapeReshape!conv2d_5/CRelu/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????b2
flatten/Reshape?
norm/mulMulflatten/Reshape:output:0flatten/Reshape:output:0*
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
	norm/Sqrty
truedivRealDivflatten/Reshape:output:0norm/Sqrt:y:0*
T0*(
_output_shapes
:??????????b2	
truediv?
IdentityIdentitytruediv:z:0 ^conv2d_3/BiasAdd/ReadVariableOp^conv2d_3/Conv2D/ReadVariableOp ^conv2d_4/BiasAdd/ReadVariableOp^conv2d_4/Conv2D/ReadVariableOp ^conv2d_5/BiasAdd/ReadVariableOp^conv2d_5/Conv2D/ReadVariableOp*
T0*(
_output_shapes
:??????????b2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????: : : : : : 2B
conv2d_3/BiasAdd/ReadVariableOpconv2d_3/BiasAdd/ReadVariableOp2@
conv2d_3/Conv2D/ReadVariableOpconv2d_3/Conv2D/ReadVariableOp2B
conv2d_4/BiasAdd/ReadVariableOpconv2d_4/BiasAdd/ReadVariableOp2@
conv2d_4/Conv2D/ReadVariableOpconv2d_4/Conv2D/ReadVariableOp2B
conv2d_5/BiasAdd/ReadVariableOpconv2d_5/BiasAdd/ReadVariableOp2@
conv2d_5/Conv2D/ReadVariableOpconv2d_5/Conv2D/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_3_layer_call_fn_13833364

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
GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_3_layer_call_and_return_conditional_losses_138328532
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
&__inference_signature_wrapper_13833109
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
#__inference__wrapped_model_138328322
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
?
?
F__inference_conv2d_4_layer_call_and_return_conditional_losses_13832873

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
?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_13833355

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
?
?
F__inference_conv2d_5_layer_call_and_return_conditional_losses_13832893

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
?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_13832853

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
+__inference_conv2d_4_layer_call_fn_13833387

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
GPU2*0,1J 8? *O
fJRH
F__inference_conv2d_4_layer_call_and_return_conditional_losses_138328732
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
?
?
)__inference_critic_layer_call_fn_13833307

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
GPU2*0,1J 8? *M
fHRF
D__inference_critic_layer_call_and_return_conditional_losses_138329132
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
?
F
*__inference_flatten_layer_call_fn_13833421

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
GPU2*0,1J 8? *N
fIRG
E__inference_flatten_layer_call_and_return_conditional_losses_138329052
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
StatefulPartitionedCall:0??????????btensorflow/serving/predict:?t
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
9_default_save_signature
*:&call_and_return_all_conditional_losses
;__call__"?
_tf_keras_model?{"name": "critic", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Critic", "config": {"layer was saved without config": true}, "is_graph_network": false, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [64, 28, 28, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Critic"}}
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
_tf_keras_layer?	{"name": "conv2d_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_3", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "crelu_v2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 0}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 1}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 2, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 3}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*>&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2d_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_4", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "crelu_v2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 7}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 64]}}
?


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
*@&call_and_return_all_conditional_losses
A__call__"?	
_tf_keras_layer?	{"name": "conv2d_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d_5", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "crelu_v2", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 8}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 9}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 10, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}, "shared_object_id": 11}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 128]}}
?
trainable_variables
regularization_losses
	variables
	keras_api
*B&call_and_return_all_conditional_losses
C__call__"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 13}}
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
trainable_variables

 layers
!layer_metrics
"layer_regularization_losses
#metrics
regularization_losses
$non_trainable_variables
	variables
;__call__
9_default_save_signature
*:&call_and_return_all_conditional_losses
&:"call_and_return_conditional_losses"
_generic_user_object
,
Dserving_default"
signature_map
):' 2conv2d_3/kernel
: 2conv2d_3/bias
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
&layer_metrics
trainable_variables
'layer_regularization_losses
(metrics
regularization_losses
)non_trainable_variables
	variables
=__call__
*<&call_and_return_all_conditional_losses
&<"call_and_return_conditional_losses"
_generic_user_object
):'@@2conv2d_4/kernel
:@2conv2d_4/bias
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
+layer_metrics
trainable_variables
,layer_regularization_losses
-metrics
regularization_losses
.non_trainable_variables
	variables
?__call__
*>&call_and_return_all_conditional_losses
&>"call_and_return_conditional_losses"
_generic_user_object
+:)??2conv2d_5/kernel
:?2conv2d_5/bias
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
0layer_metrics
trainable_variables
1layer_regularization_losses
2metrics
regularization_losses
3non_trainable_variables
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
5layer_metrics
trainable_variables
6layer_regularization_losses
7metrics
regularization_losses
8non_trainable_variables
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
?2?
#__inference__wrapped_model_13832832?
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
?2?
D__inference_critic_layer_call_and_return_conditional_losses_13833150
D__inference_critic_layer_call_and_return_conditional_losses_13833191
D__inference_critic_layer_call_and_return_conditional_losses_13833232
D__inference_critic_layer_call_and_return_conditional_losses_13833273?
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
)__inference_critic_layer_call_fn_13833290
)__inference_critic_layer_call_fn_13833307
)__inference_critic_layer_call_fn_13833324
)__inference_critic_layer_call_fn_13833341?
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
F__inference_conv2d_3_layer_call_and_return_conditional_losses_13833355?
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
+__inference_conv2d_3_layer_call_fn_13833364?
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
F__inference_conv2d_4_layer_call_and_return_conditional_losses_13833378?
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
+__inference_conv2d_4_layer_call_fn_13833387?
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
F__inference_conv2d_5_layer_call_and_return_conditional_losses_13833401?
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
+__inference_conv2d_5_layer_call_fn_13833410?
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
E__inference_flatten_layer_call_and_return_conditional_losses_13833416?
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
*__inference_flatten_layer_call_fn_13833421?
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
&__inference_signature_wrapper_13833109input_1"?
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
#__inference__wrapped_model_13832832x
8?5
.?+
)?&
input_1?????????
? "4?1
/
output_1#? 
output_1??????????b?
F__inference_conv2d_3_layer_call_and_return_conditional_losses_13833355l
7?4
-?*
(?%
inputs?????????
? "-?*
#? 
0?????????@
? ?
+__inference_conv2d_3_layer_call_fn_13833364_
7?4
-?*
(?%
inputs?????????
? " ??????????@?
F__inference_conv2d_4_layer_call_and_return_conditional_losses_13833378m7?4
-?*
(?%
inputs?????????@
? ".?+
$?!
0??????????
? ?
+__inference_conv2d_4_layer_call_fn_13833387`7?4
-?*
(?%
inputs?????????@
? "!????????????
F__inference_conv2d_5_layer_call_and_return_conditional_losses_13833401n8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_conv2d_5_layer_call_fn_13833410a8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_critic_layer_call_and_return_conditional_losses_13833150m
;?8
1?.
(?%
inputs?????????
p 
? "&?#
?
0??????????b
? ?
D__inference_critic_layer_call_and_return_conditional_losses_13833191m
;?8
1?.
(?%
inputs?????????
p
? "&?#
?
0??????????b
? ?
D__inference_critic_layer_call_and_return_conditional_losses_13833232n
<?9
2?/
)?&
input_1?????????
p 
? "&?#
?
0??????????b
? ?
D__inference_critic_layer_call_and_return_conditional_losses_13833273n
<?9
2?/
)?&
input_1?????????
p
? "&?#
?
0??????????b
? ?
)__inference_critic_layer_call_fn_13833290a
<?9
2?/
)?&
input_1?????????
p 
? "???????????b?
)__inference_critic_layer_call_fn_13833307`
;?8
1?.
(?%
inputs?????????
p 
? "???????????b?
)__inference_critic_layer_call_fn_13833324`
;?8
1?.
(?%
inputs?????????
p
? "???????????b?
)__inference_critic_layer_call_fn_13833341a
<?9
2?/
)?&
input_1?????????
p
? "???????????b?
E__inference_flatten_layer_call_and_return_conditional_losses_13833416b8?5
.?+
)?&
inputs??????????
? "&?#
?
0??????????b
? ?
*__inference_flatten_layer_call_fn_13833421U8?5
.?+
)?&
inputs??????????
? "???????????b?
&__inference_signature_wrapper_13833109?
C?@
? 
9?6
4
input_1)?&
input_1?????????"4?1
/
output_1#? 
output_1??????????b