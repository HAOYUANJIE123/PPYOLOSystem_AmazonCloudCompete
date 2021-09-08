/**
 * Created by 30947 on 2018/8/10.
 */

    //警员变量
var noice=[{id:'1001',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员一',type:'巡查'  ,stol:'1',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.586691,coordY:  25.890847,type1:"在线",tit:"电话",action:"4",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1002',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员二',type:'处警',stol:'2',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.620036,coordY:  25.849234,type1:"在线",tit:"电话",action:"4",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1003',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员三',type:'处警',stol:'3',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.582092,coordY:  25.796157,type1:"在线",tit:"电话",action:"4",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1004',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员四',type:'待命',stol:'1',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.526691,coordY:  25.890807,type1:"在线",tit:"电话",action:"4",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1005',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员五',type:'待命',stol:'2',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.583691,coordY:  25.880847,type1:"在线",tit:"电话",action:"4",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1006',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员六',type:'巡查',stol:'3',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.586671,coordY:  25.890827,type1:"在线",tit:"电话",action:"4",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"}];

var noice1=[{id:'1001',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员一',type:'巡查'  ,stol:'1',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.645333,coordY:  25.998969,type1:"在线",tit:"电话",action:"语音呼叫",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1002',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员二',type:'处警',stol:'2',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.674078,coordY:  25.883565,type1:"在线",tit:"电话",action:"语音呼叫",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1003',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员三',type:'处警',stol:'3',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.685577,coordY:  25.812811,type1:"在线",tit:"电话",action:"语音呼叫",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1004',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员四',type:'待命',stol:'1',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.672928,coordY:  25.729516,type1:"在线",tit:"电话",action:"语音呼叫",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1005',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员五',type:'待命',stol:'2',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.712023,coordY:  25.673258,type1:"在线",tit:"电话",action:"语音呼叫",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"},
    {id:'1006',top:'警员信息',idtit:'警员编号',nametit:'警员姓名',unittit:'所属单位',valtit:'处警单号',timetit:'处警时间',time:'2018-09-08',notetit:'携带装备',note:'警棍、对讲机、警务车辆（贵K·00000）、执法记录仪 、',typetit:'状态',name:'巡逻警员六',type:'巡查',stol:'3',address:'独山县麻万镇78号',unit:'县公安局',val:"13516123567",coordX:107.758016,coordY:  25.608633,type1:"在线",tit:"电话",action:"语音呼叫",action1:"指令下达",action2:"轨迹回放",action3:"视频图像"}];
//监控设备变量
var set=[{id:'1001',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备一',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"良好" ,stol:'1',coordX:107.586691,coordY:  25.890841,type1:"在线",tit:"",action:"",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1002',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备二',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"良好" ,stol:'1',coordX:107.620035,coordY:  25.849233,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1003',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备三',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"维修" ,stol:'1',coordX:107.582090,coordY:  25.796167,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1004',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备四',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"很好" ,stol:'1',coordX:107.526681,coordY:  25.890817,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1005',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备五',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"很好" ,stol:'1',coordX:107.583671,coordY:  25.880827,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1006',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备六',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"良好" ,stol:'1',coordX:107.586651,coordY:  25.890837,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"}];

var set1=[{id:'1001',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备一',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"良好" ,stol:'1',coordX:107.544435,coordY:  25.917887,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1002',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备二',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"良好" ,stol:'1',coordX:107.611987,coordY:  25.853916,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1003',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备三',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"维修" ,stol:'1',coordX:107.595315,coordY:  25.915807,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1004',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备四',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"很好" ,stol:'1',coordX:107.553059,coordY:  25.883045,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1005',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备五',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"很好" ,stol:'1',coordX:107.568294,coordY:  25.867181,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1006',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'设备状态良好',typetit:'设备类型',name:'监控设备六',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"良好" ,stol:'1',coordX:107.587266,coordY:  25.884346,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"}];
//卡口变量
var kakou=[{id:'1001',top:'卡口信息',idtit:'卡口编号',nametit:'卡口名称',unittit:'所属单位',valtit:'状态',timetit:'运行时间',time:'2018-09-08',notetit:'备注',note:'该卡口为重要卡口',typetit:'类型',name:'卡口一',type:'卡口',address:'独山县基长镇78号',unit:'县公安局',val:"开通" ,stol:'1',coordX:107.517701,coordY:  25.769089,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1002',top:'卡口信息',idtit:'卡口编号',nametit:'卡口名称',unittit:'所属单位',valtit:'状态',timetit:'运行时间',time:'2018-09-08',notetit:'备注',note:'该卡口为重要卡口',typetit:'类型',name:'卡口二',type:'卡口',address:'独山县基长镇78号',unit:'县公安局',val:"开通" ,stol:'1',coordX:107.684427,coordY:  25.797198,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1003',top:'卡口信息',idtit:'卡口编号',nametit:'卡口名称',unittit:'所属单位',valtit:'状态',timetit:'运行时间',time:'2018-09-08',notetit:'备注',note:'该卡口为重要卡口',typetit:'类型',name:'卡口三',type:'卡口',address:'独山县基长镇78号',unit:'县公安局',val:"关闭" ,stol:'1',coordX:107.785612,coordY:  25.697223,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1004',top:'卡口信息',idtit:'卡口编号',nametit:'卡口名称',unittit:'所属单位',valtit:'状态',timetit:'运行时间',time:'2018-09-08',notetit:'备注',note:'该卡口为重要卡口',typetit:'类型',name:'卡口四',type:'卡口',address:'独山县基长镇78号',unit:'县公安局',val:"关闭" ,stol:'1',coordX:107.740769,coordY:  25.729510,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1005',top:'卡口信息',idtit:'卡口编号',nametit:'卡口名称',unittit:'所属单位',valtit:'状态',timetit:'运行时间',time:'2018-09-08',notetit:'备注',note:'该卡口为重要卡口',typetit:'类型',name:'卡口五',type:'卡口',address:'独山县基长镇78号',unit:'县公安局',val:"关闭" ,stol:'1',coordX:107.740759,coordY:  25.729516,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1006',top:'卡口信息',idtit:'卡口编号',nametit:'卡口名称',unittit:'所属单位',valtit:'状态',timetit:'运行时间',time:'2018-09-08',notetit:'备注',note:'该卡口为重要卡口',typetit:'类型',name:'卡口六',type:'卡口',address:'独山县基长镇78号',unit:'县公安局',val:"开通" ,stol:'1',coordX:107.740768,coordY:  25.729511,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"}];
//巡逻车变量
var car01=[{id:'1001',top:'车辆信息',idtit:'车辆编号',nametit:'车牌号',unittit:'所属单位',valtit:'状态',timetit:'购买时间',time:'2018-09-08',notetit:'备注',note:'该车型号为大众宝来',typetit:'类型',name:'贵JSC097',type:'巡逻车',stol:'1',address:'独山县基长镇78号',unit:'县公安局',val:"良好",coordX:107.517701,coordY:  25.769089,type1:"在线",tit:"",action:"1",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1002',top:'车辆信息',idtit:'车辆编号',nametit:'车牌号',unittit:'所属单位',valtit:'状态',timetit:'购买时间',time:'2018-09-08',notetit:'备注',note:'该车型号为大众宝来',typetit:'类型',name:'贵JSC666',type:'巡逻车',stol:'2',address:'独山县基长镇78号',unit:'县公安局',val:"良好",coordX:107.743068,coordY:   25.853396,type1:"在线",tit:"",action:"1",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1003',top:'车辆信息',idtit:'车辆编号',nametit:'车牌号',unittit:'所属单位',valtit:'状态',timetit:'购买时间',time:'2018-09-08',notetit:'备注',note:'该车型号为大众宝来',typetit:'类型',name:'贵JSC999',type:'巡逻车',stol:'3',address:'独山县基长镇78号',unit:'县公安局',val:"良好",coordX:107.716622,coordY:   25.763882,type1:"在线",tit:"",action:"1",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1004',top:'车辆信息',idtit:'车辆编号',nametit:'车牌号',unittit:'所属单位',valtit:'状态',timetit:'购买时间',time:'2018-09-08',notetit:'备注',note:'该车型号为大众宝来',typetit:'类型',name:'贵JSC356',type:'巡逻车',stol:'1',address:'独山县基长镇78号',unit:'县公安局',val:"良好",coordX:107.590141,coordY:  25.848194,type1:"在线",tit:"",action:"1",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1005',top:'车辆信息',idtit:'车辆编号',nametit:'车牌号',unittit:'所属单位',valtit:'状态',timetit:'购买时间',time:'2018-09-08',notetit:'备注',note:'该车型号为大众宝来',typetit:'类型',name:'贵JSC087',type:'巡逻车',stol:'2',address:'独山县基长镇78号',unit:'县公安局',val:"良好",coordX:107.748817,coordY:  25.66909,type1:"在线",tit:"",action:"1",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1006',top:'车辆信息',idtit:'车辆编号',nametit:'车牌号',unittit:'所属单位',valtit:'状态',timetit:'购买时间',time:'2018-09-08',notetit:'备注',note:'该车型号为大众宝来',typetit:'类型',name:'贵JSC008',type:'巡逻车',stol:'3',address:'独山县基长镇78号',unit:'县公安局',val:"良好",coordX:107.770664,coordY:   25.672216,type1:"在线",tit:"",action:"1",action1:"",action2:"预览",action3:"视频回放"}];

//移动布球
var ball=[{id:'1001',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'该设备可以移动拍摄视频',typetit:'类型',name:'移动布控球一',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局',val:"良好" ,stol:'1',coordX:107.441812,coordY:  25.9106071,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1002',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'该设备可以移动拍摄视频',typetit:'类型',name:'移动布控球二',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局' ,stol:'1',val:"",coordX:107.622336,coordY:  25.847153,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1003',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'该设备可以移动拍摄视频',typetit:'类型',name:'移动布控球三',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局' ,stol:'1',val:"",coordX:107.610838,coordY:  25.721183,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1004',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'该设备可以移动拍摄视频',typetit:'类型',name:'移动布控球四',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局' ,stol:'1',val:"",coordX:107.676378,coordY:  25.745138,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1005',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'该设备可以移动拍摄视频',typetit:'类型',name:'移动布控球五',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局' ,stol:'1',val:"",coordX:107.536099,coordY:  25.720142,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"},
    {id:'1006',top:'设备信息',idtit:'设备编号',nametit:'设备名称',unittit:'所属单位',valtit:'状态',timetit:'安装时间',time:'2018-09-08',notetit:'备注',note:'该设备可以移动拍摄视频',typetit:'类型',name:'移动布控球六',type:'监控设备',address:'独山县麻万镇78号',unit:'县公安局' ,stol:'1',val:"",coordX:107.586651,coordY:  25.890837,type1:"在线",tit:"",action:"2",action1:"",action2:"预览",action3:"视频回放"}];
var site=[{id:'1001',type1:'防控圈01',name:'防控圈01',address:'堆龙区XXXX路78号',unit:'县公安局',coordX:107.563695,coordY: 25.911647,coordX1:107.72927,coordY1:  25.767006,coordX2:107.337178,coordY2: 25.826341,coordX3:107.434913,coordY3:25.919966,psx:107.571743,psy:25.881485}


]



var scrvideo="img/videoLine.png";
var nocset=["../images/set01.png"];
var kaIcon=["../images/ka.png"];
var perIcon=["../images/tel01.png","../images/tel02.png","../images/tel03.png"];
var carIcon=["../images/car01.png","../images/car02.png","../images/car03.png"];

var ballIcon=["../images/ball.png"];
function showCamera(setNum,srcVal,flag,s){
    for( i=0;i<setNum.length;i++){
        var pt = new BMap.Point(setNum[i].coordX,setNum[i].coordY);
        var myIcon;
        if(setNum[i].stol==1){

             myIcon = new BMap.Icon(srcVal[0], new BMap.Size(32,32));
        }
        else if(setNum[i].stol==2){
             myIcon = new BMap.Icon(srcVal[1], new BMap.Size(32,32));
        }
        else{
             myIcon = new BMap.Icon(srcVal[2], new BMap.Size(32,32));
        }

        var mak2= new BMap.Marker(pt,{icon:myIcon});
        var  label = new BMap.Label(setNum[i].name,{offset:new BMap.Size(20,-30)});
        label.setStyle({

            color: '#fff',
            background:"url(../images/ico_level_06.png) no-repeat",
            backgruoundSize:"cover",
            width:"120px",
            height:"29px",
            textAlign: "center",
            lineHeight: "20px",
            border:0

        });
        mak2.setLabel(label);
        //制作a标签
        function getAelemt(a){
            if(a==4){
                return "<a href='#' style='background-color: #2d7197;color: #FFFFff;width:80px;height:22px;line-height:22px;;text-decoration:none;border-radius:10px;display: inline-block;font-size: 12px;margin-right: 8px;' id='a1'>语音呼叫</a>"+
                    "<a href='#'style='background-color: #2d7197;color: #FFFFff;width:80px;height:22px;line-height:22px;;text-decoration:none;border-radius:10px;display: inline-block;font-size: 12px;margin-right: 8px;'id='a2'>指令下达</a>"+
                    "<a href='#'style='background-color: #2d7197;color: #FFFFff;width:80px;height:22px;line-height:22px;;text-decoration:none;border-radius:10px;display: inline-block;font-size: 12px;margin-right: 8px;' id='a3'>轨迹回话</a>"+
                    "<a href='#' style='background-color: #2d7197;color: #FFFFff;width:80px;height:22px;line-height:22px;;text-decoration:none;border-radius:10px;display: inline-block;font-size: 12px;margin-right: 8px;' id='a4'>视频图像</a>";
            }
            else if(a==2){
                return "<a href='#' style='background-color: #2d7197;color: #FFFFff;width:80px;height:22px;line-height:22px;;text-decoration:none;border-radius:10px;display: inline-block;font-size: 12px;margin-right: 8px;' id='a1'>预览</a>"+

                    "<a href='#' style='background-color: #2d7197;color: #FFFFff;width:80px;height:22px;line-height:22px;;text-decoration:none;border-radius:10px;display: inline-block;font-size: 12px;margin-right: 8px;' id='a4'>视频回放</a>";
            }
            else{
                return "<a href='#' style='background-color: #2d7197;color: #FFFFff;width:80px;height:22px;line-height:22px;;text-decoration:none;border-radius:10px;display: inline-block;font-size: 12px;margin-right: 8px;' id='a4'>轨迹查看</a>";
            }
        }
        var sContent1 =
   " <div style='width: 500px;background-color: #fff;z-index:999;font-size: 14px'><div style='height: 26px;color: #FFFFff;background-color: #458fb9;text-indent: 1em;line-height: 26px'>"+setNum[i].top+"</div><table style='border: 1px solid #dddddd;border-collapse: collapse;width: 98%;margin: 10px auto' border='1' >"+
            "<tr><td rowspan='3' style='width: 60px'><img src='../images/dail_ph.png' style='width: 60px'> </td><td style='font-weight: bold;color: #5b7295;text-align: right;width: 80px;'>"+setNum[i].idtit+"：</td><td>"+setNum[i].id+"</td><td style='font-weight: bold;color: #5b7295;text-align: right;width: 80px'>"+setNum[i].nametit+"：</td><td>"+setNum[i].name+"</td></tr>"+
        "<tr><td style='font-weight: bold;color: #5b7295;text-align: right'>"+setNum[i].typetit+"：</td><td>"+setNum[i].type+"</td><td style='font-weight: bold;color: #5b7295;text-align: right'>"+setNum[i].unittit+"：</td><td>"+setNum[i].unit+"</td></tr>"+
       "<tr><td style='font-weight: bold;color: #5b7295;text-align: right'>"+setNum[i].valtit+"：</td><td>"+setNum[i].val+"</td><td style='font-weight: bold;color: #5b7295;text-align: right'>"+setNum[i].timetit+"：</td><td>"+setNum[i].time+"</td></tr></table>"+
       " <div style='padding: 5px;font-weight: bold;color: #5b7295;float: left'>"+setNum[i].notetit+"：</div><div style='float: left;border:1px solid #7f9abe;border-radius: 3px;box-sizing: border-box;padding: 5px;font-size: 12px;color: #666666;width: 405px '>"+setNum[i].note+"</div>"+
           " <div style='text-align: center;padding: 5px;margin-bottom: 2px;float: left;' id='flag'>"+getAelemt(setNum[i].action)+"</div></div>";
        var infoWindow = new BMap.InfoWindow(sContent1);
        if(flag=="警员"){



        }
        else{


        }

        mak2.addEventListener("click", function(){
            this.openInfoWindow(infoWindow);})
        map.addOverlay(mak2);
    }

    map.panTo(point);
}
//不添加标注
function showCamera01(setNum,srcVal){
    for( i=0;i<setNum.length;i++){
        var pt = new BMap.Point(setNum[i].coordX,setNum[i].coordY);

        var myIcon;
        if(setNum[i].stol==1){

            myIcon = new BMap.Icon(srcVal[0], new BMap.Size(32,32));
        }
        else if(setNum[i].stol==2){
            myIcon = new BMap.Icon(srcVal[1], new BMap.Size(32,32));
        }
        else{
            myIcon = new BMap.Icon(srcVal[2], new BMap.Size(32,32));
        }


        var mak2= new BMap.Marker(pt,{icon:myIcon});




        map.addOverlay(mak2);
    }
    map.panTo(point);
}
function showSite(setNum){
    for( i=0;i<setNum.length;i++){
       var  poi = new BMap.Point(setNum[i].psx, setNum[i].psy);
        var circle = new BMap.Circle(poi,2000,{strokeColor:"red", strokeWeight:2, strokeOpacity:0.5});
        var circle1 = new BMap.Circle(poi,5000,{strokeColor:"blue", strokeWeight:2, strokeOpacity:0.5});
        map.addOverlay(circle);
        map.addOverlay(circle1);
        //map.panTo(poi);

    }

}


// 百度地图API功能
var map = new BMap.Map("map_box");    // 创建Map实例
var point = new BMap.Point(116.404, 39.915);
map.centerAndZoom(point, 11);
//添加地图类型控件
var size1 = new BMap.Size(15, 50);
map.addControl(new BMap.MapTypeControl({
    offset: size1,
    mapTypes:[
        BMAP_NORMAL_MAP,
        BMAP_HYBRID_MAP,

    ]}));

map.enableScrollWheelZoom(true);     //开启鼠标滚轮缩放

    // 随机向地图添加25个标注
    var bounds = map.getBounds();
    var sw = bounds.getSouthWest();
    var ne = bounds.getNorthEast();
    var lngSpan = Math.abs(sw.lng - ne.lng);
    var latSpan = Math.abs(ne.lat - sw.lat);
var x01;var y01;


//点击获取坐标
//function showInfo(e){
//    alert(e.point.lng + ", " + e.point.lat);
//}
//map.addEventListener("click", showInfo);

var marker = new BMap.Marker(point);  // 创建标注

var circle = new BMap.Circle(point,500,{strokeColor:"blue", strokeWeight:2, strokeOpacity:0.5});
var polyline = new BMap.Polyline([
    new BMap.Point(107.644183, 26.006243),
    new BMap.Point(107.691326, 25.831544),
    new BMap.Point(107.66143, 25.748263),
    new BMap.Point(107.768364, 25.589865),
    new BMap.Point(107.824706, 25.594036)
], {strokeColor:"blue", strokeWeight:5, strokeOpacity:0.5});   //创建折线
//添加警情信息点
var pArr=[{coordX:107.644183,coordY:  26.006243,'tit':'A'},
    {coordX:107.691326,coordY:  25.831544,'tit':'B'},
    {coordX:107.66143,coordY:  25.748263,'tit':'C'}];


var ps;
function aniPoint(){
    map.addOverlay(marker);
    marker.setAnimation(BMAP_ANIMATION_BOUNCE);
    map.addOverlay(circle);//增加多边形
    getBoundary();



}
//行政区划

function getBoundary(){
    var bdary = new BMap.Boundary();
    bdary.get("独山县", function(rs){       //获取行政区域

        var count = rs.boundaries.length; //行政区域的点有多少个
        if (count === 0) {
            alert('未能获取当前输入行政区域');
            return ;
        }
        var pointArray = [];
        for (var i = 0; i < count; i++) {
            var ply = new BMap.Polygon(rs.boundaries[i], {strokeWeight: 2, strokeColor: "#ff0000"}); //建立多边形覆盖物
            map.addOverlay(ply);  //添加覆盖物
            pointArray = pointArray.concat(ply.getPath());
        }
        map.setViewport(pointArray);    //调整视野
        addlabel();
    });
}



function addlabel() {
    var pointArray = [
        new BMap.Point(107.580942, 25.859639)
        ];
    var optsArray = [{},{},{}];
    var labelArray = [];
    var contentArray = [
        "独山县区域"
        ];
    for(var i = 0;i < pointArray.length; i++) {
        optsArray[i].position = pointArray[i];
        labelArray[i] = new BMap.Label(contentArray[i],optsArray[i]);
        labelArray[i].setStyle({
            color : "red",
            fontSize : "12px",
            height : "20px",
            lineHeight : "20px",
            fontFamily:"微软雅黑"
        });
        map.addOverlay(labelArray[i]);
    }
}
var  label01;

//添加安保信息行程
function safeLine(){
    map.clearOverlays();
    var mak2;


    map.addOverlay(polyline);
    for(i=0;i<pArr.length;i++){
        ps=new BMap.Point(pArr[i].coordX, pArr[i].coordY);
        mak2= new BMap.Marker(ps);
        label01 = new BMap.Label(pArr[i].tit,{offset:new BMap.Size(2,3)});
        mak2.setLabel(label01);
        map.addOverlay(mak2);

    }
    showCamera01(noice1,perIcon);
    showCamera01(set1,nocset);




}



var f1="警员";
var f2="监控设备";
var f3="重点卡口";
var f4="巡逻车辆";
var f5="移动布控球";
    //监控设备
$("#check02").change(function(){
    // inputCheck(point,marker1,infoWindow1);
    if($('input[type=checkbox]').is(':checked')){
        showCamera(set,nocset,f2);}
    else{
        map.clearOverlays();
    }
})
    //巡逻警员
    $("#check01").change(function(){
       // inputCheck(point,marker1,infoWindow1);
        if($('input[type=checkbox]').is(':checked')){
        showCamera(noice,perIcon,f1);
        }
        else{
            map.clearOverlays();
        }
    })
//卡口
$("#check03").change(function(){
    // inputCheck(point,marker1,infoWindow1);
    if($('input[type=checkbox]').is(':checked')){
        showCamera(kakou,kaIcon,f3);
    }
    else{
        map.clearOverlays();
    }
})
//巡逻车辆
$("#check04").change(function(){
    // inputCheck(point,marker1,infoWindow1);
    if($('input[type=checkbox]').is(':checked')){
        showCamera(car01,carIcon,f4);



    }
    else{
        map.clearOverlays();
    }
})
//移动布球
$("#check05").change(function(){
    // inputCheck(point,marker1,infoWindow1);
    if($('input[type=checkbox]').is(':checked')){
        showCamera(ball,ballIcon,f5);
    }
    else{
        map.clearOverlays();
    }
})
//防控圈01
$("#check06").change(function(){

    if($('input[type=checkbox]').is(':checked')){
        showSite(site);
        showCamera01(set1,nocset);
    }
    else{
        map.clearOverlays();
    }
})
//防控圈02
$("#check07").change(function(){

    if($('input[type=checkbox]').is(':checked')){
        showSite(site);
        showCamera01(set1,nocset);
    }
    else{
        map.clearOverlays();
    }
})



//定位
    function addLocal(x,y){


      var pt=new BMap.Point(x, y);
        var markers = new BMap.Marker(pt);
        map.addOverlay(markers);
        map.panTo(pt);
        markers.addEventListener("click",function(){
            this.openInfoWindow(infoWindow);
        })
    }
//定位1
function addLocal1(){


    var pt=new BMap.Point(107.553346, 25.826341);
    var markers = new BMap.Marker(pt);
    map.addOverlay(markers);
    map.panTo(pt);
    markers.addEventListener("click",function(){
        this.openInfoWindow(infoWindow);
    })
}


$(".search_location").click(function(){

    addLocal();
})
function findAddr(val){
    $("#add_res").show();
    $("#addr_local").hide();



    var options = {
        onSearchComplete: function(results){

            var len=results.Br.length;
            var liHtml="";
            for(i=0;i<len;i++){
                var tit=results.Br[i].title;
                var addr=results.Br[i].address;
                x01=results.Br[i].point.lng;
               y01=results.Br[i].point.lat;

               var li="<li><p><strong>"+tit+"</strong><span class='right'><a href='javascript:void(0)' title='视频'><img src='img/nav_5.png'> </a><a href='javascript:void(0)' title='定位' onclick='addLocal(x01,y01)'><img src='img/find01.png'> </a> </span></p> <p>"+addr+"</p></li>";
                console.log(li);
                liHtml+=li;

            }

            document.getElementById("addr_ul").innerHTML=liHtml;

        }
    };
   val=$("#addName").val();
    var local = new BMap.LocalSearch(map, options);
    local.search(val);
}
function findAddr1(){
    $("#addr_local li").click(function(){
        $("#add_res").show();
        $("#addr_local").hide();
        var options = {
            onSearchComplete: function(results){


                var len=results.Br.length;
                var liHtml="";
                for(i=0;i<len;i++){
                    var tit=results.Br[i].title;
                    var addr=results.Br[i].address;
                   x01=results.Br[i].point.lng;
                    y01=results.Br[i].point.lat;
                    localPoint= new BMap.Point(x01,y01);



                    var li="<li><p><strong>"+tit+"</strong><span class='right'><a href='javascript:void(0)' title='视频'><img src='img/nav_5.png'> </a><a href='javascript:void(0)' title='定位' onclick='addLocal(x01,y01)'><img src='img/find01.png'> </a> </span></p> <p>"+addr+"</p></li>";

                    liHtml+=li;

                }

                document.getElementById("addr_ul").innerHTML=liHtml;

            }
        };
      var  val=$(this).find(".img_div_text").text();
        var local = new BMap.LocalSearch(map, options);
        local.search(val);
    })

}


$(function(){
    findAddr1();
})
