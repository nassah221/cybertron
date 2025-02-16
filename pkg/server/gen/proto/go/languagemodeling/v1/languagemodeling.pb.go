// Code generated by protoc-gen-go. DO NOT EDIT.
// versions:
// 	protoc-gen-go v1.28.1
// 	protoc        (unknown)
// source: languagemodeling/v1/languagemodeling.proto

package languagemodelingv1

import (
	_ "google.golang.org/genproto/googleapis/api/annotations"
	protoreflect "google.golang.org/protobuf/reflect/protoreflect"
	protoimpl "google.golang.org/protobuf/runtime/protoimpl"
	reflect "reflect"
	sync "sync"
)

const (
	// Verify that this generated code is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(20 - protoimpl.MinVersion)
	// Verify that runtime/protoimpl is sufficiently up-to-date.
	_ = protoimpl.EnforceVersion(protoimpl.MaxVersion - 20)
)

type LanguageModelingRequest struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Input      string                      `protobuf:"bytes,1,opt,name=input,proto3" json:"input,omitempty"`
	Parameters *LanguageModelingParameters `protobuf:"bytes,2,opt,name=parameters,proto3" json:"parameters,omitempty"`
}

func (x *LanguageModelingRequest) Reset() {
	*x = LanguageModelingRequest{}
	if protoimpl.UnsafeEnabled {
		mi := &file_languagemodeling_v1_languagemodeling_proto_msgTypes[0]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *LanguageModelingRequest) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LanguageModelingRequest) ProtoMessage() {}

func (x *LanguageModelingRequest) ProtoReflect() protoreflect.Message {
	mi := &file_languagemodeling_v1_languagemodeling_proto_msgTypes[0]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LanguageModelingRequest.ProtoReflect.Descriptor instead.
func (*LanguageModelingRequest) Descriptor() ([]byte, []int) {
	return file_languagemodeling_v1_languagemodeling_proto_rawDescGZIP(), []int{0}
}

func (x *LanguageModelingRequest) GetInput() string {
	if x != nil {
		return x.Input
	}
	return ""
}

func (x *LanguageModelingRequest) GetParameters() *LanguageModelingParameters {
	if x != nil {
		return x.Parameters
	}
	return nil
}

type LanguageModelingParameters struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	K int32 `protobuf:"varint,1,opt,name=k,proto3" json:"k,omitempty"`
}

func (x *LanguageModelingParameters) Reset() {
	*x = LanguageModelingParameters{}
	if protoimpl.UnsafeEnabled {
		mi := &file_languagemodeling_v1_languagemodeling_proto_msgTypes[1]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *LanguageModelingParameters) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LanguageModelingParameters) ProtoMessage() {}

func (x *LanguageModelingParameters) ProtoReflect() protoreflect.Message {
	mi := &file_languagemodeling_v1_languagemodeling_proto_msgTypes[1]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LanguageModelingParameters.ProtoReflect.Descriptor instead.
func (*LanguageModelingParameters) Descriptor() ([]byte, []int) {
	return file_languagemodeling_v1_languagemodeling_proto_rawDescGZIP(), []int{1}
}

func (x *LanguageModelingParameters) GetK() int32 {
	if x != nil {
		return x.K
	}
	return 0
}

type Token struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Start  int32     `protobuf:"varint,1,opt,name=start,proto3" json:"start,omitempty"`
	End    int32     `protobuf:"varint,2,opt,name=end,proto3" json:"end,omitempty"`
	Words  []string  `protobuf:"bytes,3,rep,name=words,proto3" json:"words,omitempty"`
	Scores []float64 `protobuf:"fixed64,4,rep,packed,name=scores,proto3" json:"scores,omitempty"`
}

func (x *Token) Reset() {
	*x = Token{}
	if protoimpl.UnsafeEnabled {
		mi := &file_languagemodeling_v1_languagemodeling_proto_msgTypes[2]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *Token) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*Token) ProtoMessage() {}

func (x *Token) ProtoReflect() protoreflect.Message {
	mi := &file_languagemodeling_v1_languagemodeling_proto_msgTypes[2]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use Token.ProtoReflect.Descriptor instead.
func (*Token) Descriptor() ([]byte, []int) {
	return file_languagemodeling_v1_languagemodeling_proto_rawDescGZIP(), []int{2}
}

func (x *Token) GetStart() int32 {
	if x != nil {
		return x.Start
	}
	return 0
}

func (x *Token) GetEnd() int32 {
	if x != nil {
		return x.End
	}
	return 0
}

func (x *Token) GetWords() []string {
	if x != nil {
		return x.Words
	}
	return nil
}

func (x *Token) GetScores() []float64 {
	if x != nil {
		return x.Scores
	}
	return nil
}

type LanguageModelingResponse struct {
	state         protoimpl.MessageState
	sizeCache     protoimpl.SizeCache
	unknownFields protoimpl.UnknownFields

	Tokens []*Token `protobuf:"bytes,1,rep,name=tokens,proto3" json:"tokens,omitempty"`
}

func (x *LanguageModelingResponse) Reset() {
	*x = LanguageModelingResponse{}
	if protoimpl.UnsafeEnabled {
		mi := &file_languagemodeling_v1_languagemodeling_proto_msgTypes[3]
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		ms.StoreMessageInfo(mi)
	}
}

func (x *LanguageModelingResponse) String() string {
	return protoimpl.X.MessageStringOf(x)
}

func (*LanguageModelingResponse) ProtoMessage() {}

func (x *LanguageModelingResponse) ProtoReflect() protoreflect.Message {
	mi := &file_languagemodeling_v1_languagemodeling_proto_msgTypes[3]
	if protoimpl.UnsafeEnabled && x != nil {
		ms := protoimpl.X.MessageStateOf(protoimpl.Pointer(x))
		if ms.LoadMessageInfo() == nil {
			ms.StoreMessageInfo(mi)
		}
		return ms
	}
	return mi.MessageOf(x)
}

// Deprecated: Use LanguageModelingResponse.ProtoReflect.Descriptor instead.
func (*LanguageModelingResponse) Descriptor() ([]byte, []int) {
	return file_languagemodeling_v1_languagemodeling_proto_rawDescGZIP(), []int{3}
}

func (x *LanguageModelingResponse) GetTokens() []*Token {
	if x != nil {
		return x.Tokens
	}
	return nil
}

var File_languagemodeling_v1_languagemodeling_proto protoreflect.FileDescriptor

var file_languagemodeling_v1_languagemodeling_proto_rawDesc = []byte{
	0x0a, 0x2a, 0x6c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x69,
	0x6e, 0x67, 0x2f, 0x76, 0x31, 0x2f, 0x6c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x6d, 0x6f,
	0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x12, 0x13, 0x6c, 0x61,
	0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x2e, 0x76,
	0x31, 0x1a, 0x1c, 0x67, 0x6f, 0x6f, 0x67, 0x6c, 0x65, 0x2f, 0x61, 0x70, 0x69, 0x2f, 0x61, 0x6e,
	0x6e, 0x6f, 0x74, 0x61, 0x74, 0x69, 0x6f, 0x6e, 0x73, 0x2e, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x22,
	0x80, 0x01, 0x0a, 0x17, 0x4c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x4d, 0x6f, 0x64, 0x65,
	0x6c, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x71, 0x75, 0x65, 0x73, 0x74, 0x12, 0x14, 0x0a, 0x05, 0x69,
	0x6e, 0x70, 0x75, 0x74, 0x18, 0x01, 0x20, 0x01, 0x28, 0x09, 0x52, 0x05, 0x69, 0x6e, 0x70, 0x75,
	0x74, 0x12, 0x4f, 0x0a, 0x0a, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x73, 0x18,
	0x02, 0x20, 0x01, 0x28, 0x0b, 0x32, 0x2f, 0x2e, 0x6c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65,
	0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x31, 0x2e, 0x4c, 0x61, 0x6e, 0x67,
	0x75, 0x61, 0x67, 0x65, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x50, 0x61, 0x72, 0x61,
	0x6d, 0x65, 0x74, 0x65, 0x72, 0x73, 0x52, 0x0a, 0x70, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65,
	0x72, 0x73, 0x22, 0x2a, 0x0a, 0x1a, 0x4c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x4d, 0x6f,
	0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x50, 0x61, 0x72, 0x61, 0x6d, 0x65, 0x74, 0x65, 0x72, 0x73,
	0x12, 0x0c, 0x0a, 0x01, 0x6b, 0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x01, 0x6b, 0x22, 0x5d,
	0x0a, 0x05, 0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x12, 0x14, 0x0a, 0x05, 0x73, 0x74, 0x61, 0x72, 0x74,
	0x18, 0x01, 0x20, 0x01, 0x28, 0x05, 0x52, 0x05, 0x73, 0x74, 0x61, 0x72, 0x74, 0x12, 0x10, 0x0a,
	0x03, 0x65, 0x6e, 0x64, 0x18, 0x02, 0x20, 0x01, 0x28, 0x05, 0x52, 0x03, 0x65, 0x6e, 0x64, 0x12,
	0x14, 0x0a, 0x05, 0x77, 0x6f, 0x72, 0x64, 0x73, 0x18, 0x03, 0x20, 0x03, 0x28, 0x09, 0x52, 0x05,
	0x77, 0x6f, 0x72, 0x64, 0x73, 0x12, 0x16, 0x0a, 0x06, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x18,
	0x04, 0x20, 0x03, 0x28, 0x01, 0x52, 0x06, 0x73, 0x63, 0x6f, 0x72, 0x65, 0x73, 0x22, 0x4e, 0x0a,
	0x18, 0x4c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e,
	0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73, 0x65, 0x12, 0x32, 0x0a, 0x06, 0x74, 0x6f, 0x6b,
	0x65, 0x6e, 0x73, 0x18, 0x01, 0x20, 0x03, 0x28, 0x0b, 0x32, 0x1a, 0x2e, 0x6c, 0x61, 0x6e, 0x67,
	0x75, 0x61, 0x67, 0x65, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x31, 0x2e,
	0x54, 0x6f, 0x6b, 0x65, 0x6e, 0x52, 0x06, 0x74, 0x6f, 0x6b, 0x65, 0x6e, 0x73, 0x32, 0x99, 0x01,
	0x0a, 0x17, 0x4c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x69,
	0x6e, 0x67, 0x53, 0x65, 0x72, 0x76, 0x69, 0x63, 0x65, 0x12, 0x7e, 0x0a, 0x07, 0x50, 0x72, 0x65,
	0x64, 0x69, 0x63, 0x74, 0x12, 0x2c, 0x2e, 0x6c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x6d,
	0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x31, 0x2e, 0x4c, 0x61, 0x6e, 0x67, 0x75,
	0x61, 0x67, 0x65, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x71, 0x75, 0x65,
	0x73, 0x74, 0x1a, 0x2d, 0x2e, 0x6c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x6d, 0x6f, 0x64,
	0x65, 0x6c, 0x69, 0x6e, 0x67, 0x2e, 0x76, 0x31, 0x2e, 0x4c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67,
	0x65, 0x4d, 0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x52, 0x65, 0x73, 0x70, 0x6f, 0x6e, 0x73,
	0x65, 0x22, 0x16, 0x82, 0xd3, 0xe4, 0x93, 0x02, 0x10, 0x22, 0x0b, 0x2f, 0x76, 0x31, 0x2f, 0x70,
	0x72, 0x65, 0x64, 0x69, 0x63, 0x74, 0x3a, 0x01, 0x2a, 0x42, 0x58, 0x5a, 0x56, 0x67, 0x69, 0x74,
	0x68, 0x75, 0x62, 0x2e, 0x63, 0x6f, 0x6d, 0x2f, 0x6e, 0x6c, 0x70, 0x6f, 0x64, 0x79, 0x73, 0x73,
	0x65, 0x79, 0x2f, 0x63, 0x79, 0x62, 0x65, 0x72, 0x74, 0x72, 0x6f, 0x6e, 0x2f, 0x70, 0x6b, 0x67,
	0x2f, 0x73, 0x65, 0x72, 0x76, 0x65, 0x72, 0x2f, 0x61, 0x70, 0x69, 0x73, 0x2f, 0x6c, 0x61, 0x6e,
	0x67, 0x75, 0x61, 0x67, 0x65, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e, 0x67, 0x2f, 0x76, 0x31,
	0x3b, 0x6c, 0x61, 0x6e, 0x67, 0x75, 0x61, 0x67, 0x65, 0x6d, 0x6f, 0x64, 0x65, 0x6c, 0x69, 0x6e,
	0x67, 0x76, 0x31, 0x62, 0x06, 0x70, 0x72, 0x6f, 0x74, 0x6f, 0x33,
}

var (
	file_languagemodeling_v1_languagemodeling_proto_rawDescOnce sync.Once
	file_languagemodeling_v1_languagemodeling_proto_rawDescData = file_languagemodeling_v1_languagemodeling_proto_rawDesc
)

func file_languagemodeling_v1_languagemodeling_proto_rawDescGZIP() []byte {
	file_languagemodeling_v1_languagemodeling_proto_rawDescOnce.Do(func() {
		file_languagemodeling_v1_languagemodeling_proto_rawDescData = protoimpl.X.CompressGZIP(file_languagemodeling_v1_languagemodeling_proto_rawDescData)
	})
	return file_languagemodeling_v1_languagemodeling_proto_rawDescData
}

var file_languagemodeling_v1_languagemodeling_proto_msgTypes = make([]protoimpl.MessageInfo, 4)
var file_languagemodeling_v1_languagemodeling_proto_goTypes = []interface{}{
	(*LanguageModelingRequest)(nil),    // 0: languagemodeling.v1.LanguageModelingRequest
	(*LanguageModelingParameters)(nil), // 1: languagemodeling.v1.LanguageModelingParameters
	(*Token)(nil),                      // 2: languagemodeling.v1.Token
	(*LanguageModelingResponse)(nil),   // 3: languagemodeling.v1.LanguageModelingResponse
}
var file_languagemodeling_v1_languagemodeling_proto_depIdxs = []int32{
	1, // 0: languagemodeling.v1.LanguageModelingRequest.parameters:type_name -> languagemodeling.v1.LanguageModelingParameters
	2, // 1: languagemodeling.v1.LanguageModelingResponse.tokens:type_name -> languagemodeling.v1.Token
	0, // 2: languagemodeling.v1.LanguageModelingService.Predict:input_type -> languagemodeling.v1.LanguageModelingRequest
	3, // 3: languagemodeling.v1.LanguageModelingService.Predict:output_type -> languagemodeling.v1.LanguageModelingResponse
	3, // [3:4] is the sub-list for method output_type
	2, // [2:3] is the sub-list for method input_type
	2, // [2:2] is the sub-list for extension type_name
	2, // [2:2] is the sub-list for extension extendee
	0, // [0:2] is the sub-list for field type_name
}

func init() { file_languagemodeling_v1_languagemodeling_proto_init() }
func file_languagemodeling_v1_languagemodeling_proto_init() {
	if File_languagemodeling_v1_languagemodeling_proto != nil {
		return
	}
	if !protoimpl.UnsafeEnabled {
		file_languagemodeling_v1_languagemodeling_proto_msgTypes[0].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*LanguageModelingRequest); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_languagemodeling_v1_languagemodeling_proto_msgTypes[1].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*LanguageModelingParameters); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_languagemodeling_v1_languagemodeling_proto_msgTypes[2].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*Token); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
		file_languagemodeling_v1_languagemodeling_proto_msgTypes[3].Exporter = func(v interface{}, i int) interface{} {
			switch v := v.(*LanguageModelingResponse); i {
			case 0:
				return &v.state
			case 1:
				return &v.sizeCache
			case 2:
				return &v.unknownFields
			default:
				return nil
			}
		}
	}
	type x struct{}
	out := protoimpl.TypeBuilder{
		File: protoimpl.DescBuilder{
			GoPackagePath: reflect.TypeOf(x{}).PkgPath(),
			RawDescriptor: file_languagemodeling_v1_languagemodeling_proto_rawDesc,
			NumEnums:      0,
			NumMessages:   4,
			NumExtensions: 0,
			NumServices:   1,
		},
		GoTypes:           file_languagemodeling_v1_languagemodeling_proto_goTypes,
		DependencyIndexes: file_languagemodeling_v1_languagemodeling_proto_depIdxs,
		MessageInfos:      file_languagemodeling_v1_languagemodeling_proto_msgTypes,
	}.Build()
	File_languagemodeling_v1_languagemodeling_proto = out.File
	file_languagemodeling_v1_languagemodeling_proto_rawDesc = nil
	file_languagemodeling_v1_languagemodeling_proto_goTypes = nil
	file_languagemodeling_v1_languagemodeling_proto_depIdxs = nil
}
