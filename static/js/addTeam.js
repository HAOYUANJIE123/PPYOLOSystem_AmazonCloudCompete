/**
 * Created by Administrator on 2016/8/4.
 */
var setting = {
    view: {
        dblClickExpand: false
    },
    check: {
        enable: true
    },
    callback: {
        onRightClick: OnRightClick,
        onClick:OnClick
    }
};
var zNodes =[
    {	rid:101	,	id:	10	,pId:101,name:"	贵州省",open:true,nocheck:true,
        children:[
            {id:10, name:"黔南州", open:true, noR:true,nocheck:true,
                children:[
                    {id:100, name:"独山县", noR:true, open:true,nocheck:true,
                        children:[
                            {id:1001, name:"百泉镇", noR:true, open:false,nocheck:true,
                                children:[
                                    {id:10011, name:"中华社区", noR:true, open:true,nocheck:true},
                                    {id:10012, name:"北集社区", noR:true, open:true,nocheck:true},
                                    {id:10013, name:"中南社区", noR:true, open:true,nocheck:true},
                                    {id:10014, name:"铁路社区", noR:true, open:true,nocheck:true},
                                    {id:10015, name:"和平社区", noR:true, open:true,nocheck:true},
                                    {id:10016, name:"河亭村", noR:true, open:true,nocheck:true},
                                    {id:10017, name:"双桥村", noR:true, open:true,nocheck:true},
                                    {id:10018, name:"城东村", noR:true, open:true,nocheck:true},
                                    {id:10019, name:"井桥村", noR:true, open:true,nocheck:true},
                                    {id:100010, name:"城西村", noR:true, open:true,nocheck:true},
                                    {id:100011, name:"五里村", noR:true, open:true,nocheck:true},
                                    {id:100012, name:"新民村", noR:true, open:true,nocheck:true},
                                    {id:100013, name:"摆独村", noR:true, open:true,nocheck:true},
                                    {id:100014, name:"其山村", noR:true, open:true,nocheck:true},
                                    {id:100015, name:"里腊村", noR:true, open:true,nocheck:true},
                                    {id:100016, name:"凤汝村", noR:true, open:true,nocheck:true},
                                    {id:100017, name:"尧梭村", noR:true, open:true,nocheck:true},
                                    {id:100018, name:"大河村", noR:true, open:true,nocheck:true},
                                    {id:100019, name:"拉偶村", noR:true, open:true,nocheck:true},
                                    {id:100020, name:"新力村", noR:true, open:true,nocheck:true},
                                    {id:100021, name:"羊凤村", noR:true, open:true,nocheck:true}
                                ]},
                            {id:1002, name:"麻尾镇", noR:true, open:false,nocheck:true,
                                children:[
                                    {id:10021, name:"火车站社区", noR:true, open:true,nocheck:true},
                                    {id:10022, name:"第一社区", noR:true, open:true,nocheck:true},
                                    {id:10023, name:"麻道村", noR:true, open:true,nocheck:true},
                                    {id:10024, name:"尧拉村", noR:true, open:true,nocheck:true},
                                    {id:10025, name:"和平村", noR:true, open:true,nocheck:true},
                                    {id:10026, name:"南门村", noR:true, open:true,nocheck:true},
                                    {id:10027, name:"麻尾村", noR:true, open:true,nocheck:true},
                                    {id:10028, name:"塘香村", noR:true, open:true,nocheck:true},
                                    {id:10029, name:"董改村", noR:true, open:true,nocheck:true},
                                    {id:100210, name:"甲腾村", noR:true, open:true,nocheck:true},
                                    {id:100211, name:"新尧村", noR:true, open:true,nocheck:true},
                                    {id:100212, name:"场坝村", noR:true, open:true,nocheck:true},
                                    {id:100213, name:"三棒村", noR:true, open:true,nocheck:true},
                                    {id:100214, name:"泗亭村", noR:true, open:true,nocheck:true},
                                    {id:100215, name:"播纳村", noR:true, open:true,nocheck:true},
                                    {id:100216, name:"麻寨村", noR:true, open:true,nocheck:true},
                                    {id:100217, name:"舒家坪村", noR:true, open:true,nocheck:true},
                                    {id:100218, name:"黄后村", noR:true, open:true,nocheck:true},
                                    {id:100219, name:"尧王村", noR:true, open:true,nocheck:true},
                                    {id:100220, name:"坝望村", noR:true, open:true,nocheck:true},
                                    {id:100221, name:"尧董村", noR:true, open:true,nocheck:true}
                                ]
                            },
                            {id:1003, name:"影山镇", noR:true, open:false,nocheck:true,
                                children:[
                                    {id:10031, name:"班台村", noR:true, open:true,nocheck:true},
                                    {id:10032, name:"苗翁村", noR:true, open:true,nocheck:true},
                                    {id:10033, name:"甲乙村", noR:true, open:true,nocheck:true},
                                    {id:10034, name:"甲定村", noR:true, open:true,nocheck:true},
                                    {id:10035, name:"甲西村", noR:true, open:true,nocheck:true},
                                    {id:10036, name:"达头村", noR:true, open:true,nocheck:true},
                                    {id:10037, name:"新联村", noR:true, open:true,nocheck:true},
                                    {id:10038, name:"紫林山村", noR:true, open:true,nocheck:true},
                                    {id:10039, name:"东泥村", noR:true, open:true,nocheck:true},
                                    {id:100310, name:"黄桥村", noR:true, open:true,nocheck:true},
                                    {id:100311, name:"石板村", noR:true, open:true,nocheck:true},
                                    {id:100312, name:"翁奇村", noR:true, open:true,nocheck:true}
                                ]},
                            {id:1004, name:"基长镇", noR:true, open:false,nocheck:true,
                                children:[
                                    {id:10041, name:"基长社区", noR:true, open:true,nocheck:true},
                                    {id:10042, name:"麻董村", noR:true, open:true,nocheck:true},
                                    {id:10043, name:"狮山村", noR:true, open:true,nocheck:true},
                                    {id:10044, name:"高塘村", noR:true, open:true,nocheck:true},
                                    {id:10045, name:"羊坪村", noR:true, open:true,nocheck:true},
                                    {id:10046, name:"开罗村", noR:true, open:true,nocheck:true},
                                    {id:10047, name:"董秧村", noR:true, open:true,nocheck:true},
                                    {id:10048, name:"兴合村", noR:true, open:true,nocheck:true},
                                    {id:10049, name:"尧新村", noR:true, open:true,nocheck:true},
                                    {id:100410, name:"水岩村", noR:true, open:true,nocheck:true},
                                    {id:100411, name:"阳地村", noR:true, open:true,nocheck:true}
                                ]},
                            {id:1005, name:"玉水镇", noR:true, open:false,nocheck:true,
                                children:[
                                    {id:10051, name:"本寨村", noR:true, open:true,nocheck:true},
                                    {id:10052, name:"塘联村", noR:true, open:true,nocheck:true},
                                    {id:10053, name:"黎罗村", noR:true, open:true,nocheck:true},
                                    {id:10054, name:"龙点村", noR:true, open:true,nocheck:true},
                                    {id:10055, name:"新和村", noR:true, open:true,nocheck:true},
                                    {id:10056, name:"丙约村", noR:true, open:true,nocheck:true},
                                    {id:10057, name:"拉芒村", noR:true, open:true,nocheck:true},
                                    {id:10058, name:"丙高村", noR:true, open:true,nocheck:true},
                                    {id:10059, name:"温泉村", noR:true, open:true,nocheck:true},
                                    {id:100510, name:"林桥村", noR:true, open:true,nocheck:true},
                                    {id:100511, name:"双星村", noR:true, open:true,nocheck:true},
                                    {id:100512, name:"群力村", noR:true, open:true,nocheck:true}
                                ]},
                            {id:1006, name:"上司镇", noR:true, open:false,nocheck:true,
                                children:[
                                    {id:10061, name:"上司村", noR:true, open:true,nocheck:true},
                                    {id:10062, name:"拉么村", noR:true, open:true,nocheck:true},
                                    {id:10063, name:"仁等村", noR:true, open:true,nocheck:true},
                                    {id:10064, name:"交摆村", noR:true, open:true,nocheck:true},
                                    {id:10065, name:"拉旺村", noR:true, open:true,nocheck:true},
                                    {id:10066, name:"打或村", noR:true, open:true,nocheck:true},
                                    {id:10067, name:"盖寨村", noR:true, open:true,nocheck:true},
                                    {id:10068, name:"袍寨村", noR:true, open:true,nocheck:true},
                                    {id:10069, name:"筹洞村", noR:true, open:true,nocheck:true},
                                    {id:100610, name:"王龙村", noR:true, open:true,nocheck:true},
                                    {id:100611, name:"屯脚村", noR:true, open:true,nocheck:true},
                                    {id:100612, name:"峰洞村", noR:true, open:true,nocheck:true},
                                    {id:100613, name:"学庄村", noR:true, open:true,nocheck:true},
                                    {id:100614, name:"龙泉村", noR:true, open:true,nocheck:true}
                                ]},
                            {id:1007, name:"下司镇", noR:true, open:false,nocheck:true,
                                children:[
                                    {id:10071, name:"星朗村", noR:true, open:true,nocheck:true},
                                    {id:10072, name:"四九村", noR:true, open:true,nocheck:true},
                                    {id:10073, name:"下司村", noR:true, open:true,nocheck:true},
                                    {id:10074, name:"塘芒村", noR:true, open:true,nocheck:true},
                                    {id:10075, name:"阳角村", noR:true, open:true,nocheck:true},
                                    {id:10076, name:"麻根村", noR:true, open:true,nocheck:true},
                                    {id:10077, name:"拉抹村", noR:true, open:true,nocheck:true},
                                    {id:10078, name:"拉播村", noR:true, open:true,nocheck:true},
                                    {id:10079, name:"新和村", noR:true, open:true,nocheck:true},
                                    {id:100710, name:"群巷村", noR:true, open:true,nocheck:true},
                                    {id:100711, name:"新兴村", noR:true, open:true,nocheck:true},
                                    {id:100712, name:"新同村", noR:true, open:true,nocheck:true}
                                ]},
                            {id:1008, name:"麻万镇", noR:true, open:false,nocheck:true,
                                children:[
                                    {id:10081, name:"麻抹村", noR:true, open:true,nocheck:true},
                                    {id:10082, name:"麻万村", noR:true, open:true,nocheck:true},
                                    {id:10083, name:"三里村", noR:true, open:true,nocheck:true},
                                    {id:10084, name:"丙志村", noR:true, open:true,nocheck:true},
                                    {id:10085, name:"甲摆村", noR:true, open:true,nocheck:true},
                                    {id:10086, name:"迎春村", noR:true, open:true,nocheck:true},
                                    {id:10087, name:"拉林村", noR:true, open:true,nocheck:true},
                                    {id:10088, name:"花园村", noR:true, open:true,nocheck:true},
                                    {id:10089, name:"铜鼓村", noR:true, open:true,nocheck:true},
                                    {id:100810, name:"石牛村", noR:true, open:true,nocheck:true}
                                ]},
                            {id:1009, name:"尧梭乡", noR:true, open:true,nocheck:false},
                            {id:1002, name:"羊凤乡", noR:true, open:true,nocheck:false},
                            {id:1010, name:"甲定水族乡", noR:true, open:true,nocheck:false},
                            {id:1011, name:"翁台水族乡", noR:true, open:true,nocheck:false},
                            {id:1012, name:"本寨水族乡", noR:true, open:true,nocheck:false},
                            {id:1013, name:"水岩乡", noR:true, open:true,nocheck:false},
                            {id:1014, name:"打羊乡", noR:true, open:true,nocheck:false},
                            {id:1015, name:"尧棒乡", noR:true, open:true,nocheck:false},
                            {id:1016, name:"黄后乡", noR:true, open:true,nocheck:false},
                            {id:1017, name:"董岭乡", noR:true, open:true,nocheck:false}
                        ]
                    }
                ]}
        ]
    }
];;
function OnRightClick(event, treeId, treeNode) {
    if (!treeNode && event.target.tagName.toLowerCase() != "button" && $(event.target).parents("a").length == 0) {
        zTree.cancelSelectedNode();
        //showRMenu("root", event.clientX, event.clientY);
    } else if (treeNode && !treeNode.noR) {
        zTree.selectNode(treeNode);
        //showRMenu("node", event.clientX, event.clientY);
    }
}
function showRMenu(type, x, y) {
    $("#rMenu ul").show();
    if (type=="root") {
        $("#m_del").hide();
        $("#m_check").hide();
        $("#m_unCheck").hide();
    } else {
        $("#m_del").show();
        $("#m_check").show();
        $("#m_unCheck").show();
    }
    rMenu.css({"top":y+"px", "left":x+"px", "visibility":"visible"});

    $("body").bind("mousedown", onBodyMouseDown);
}
function hideRMenu() {
    if (rMenu) rMenu.css({"visibility": "hidden"});
    $("body").unbind("mousedown", onBodyMouseDown);
}
function onBodyMouseDown(event){
    if (!(event.target.id == "rMenu" || $(event.target).parents("#rMenu").length>0)) {
        rMenu.css({"visibility" : "hidden"});
    }
}
var addCount = 1;

//添加事件
        function addTreeNode(names) {
            hideRMenu();
            var newNode = { name:names + (addCount++)};
            if (zTree.getSelectedNodes()[0]) {
                newNode.checked = zTree.getSelectedNodes()[0].checked;
                zTree.addNodes(zTree.getSelectedNodes()[0], newNode);
            } else {
                zTree.addNodes(null, newNode);
            }
        }
        function removeTreeNode() {
            hideRMenu();
            var nodes = zTree.getSelectedNodes();
            if (nodes && nodes.length>0) {
                if (nodes[0].children && nodes[0].children.length > 0) {
                    var msg = "要删除的节点是父节点，如果删除将连同子节点一起删掉。\n\n请确认！";
                    if (confirm(msg)==true){
                        zTree.removeNode(nodes[0]);
                    }
                } else {
                    zTree.removeNode(nodes[0]);
                }
            }
        }
        function checkTreeNode(checked) {
            var nodes = zTree.getSelectedNodes();
            if (nodes && nodes.length>0) {
                zTree.checkNode(nodes[0], checked, true);
            }
            hideRMenu();
        }
        function resetTree() {
            hideRMenu();
            $.fn.zTree.init($("#treeDemo"), setting, zNodes);
        }
        function OnClick(event, treeId, treeNode){
            $(".dropdown_select").val(treeNode.name);
         }
var zTree, rMenu;
$(document).ready(function(){
            $.fn.zTree.init($("#treeDemo4"), setting, zNodes);
            $.fn.zTree.init($("#treeDemo1"), setting, zNodes);
            $.fn.zTree.init($("#treeDemo2"), setting, zNodes);
            $.fn.zTree.init($("#treeDemo3"), setting, zNodes);
    $.fn.zTree.init($("#treeDemo5"), setting, zNodes);
            zTree = $.fn.zTree.getZTreeObj("treeDemo");
            rMenu = $("#rMenu");
});
function table(){
    $('#table').bootstrapTable({
        method: "get",
        striped: false,
        singleSelect: false,
        url: "../json/user.json",
        dataType: "json",
        pagination: true, //分页
        pageSize: 10,
        pageNumber: 1,
        search: false, //显示搜索框
        contentType: "application/x-www-form-urlencoded",
        queryParams: null,
        columns: [
            {
                checkbox: "true",
                field: 'check',
                align: 'center',
                valign: 'middle'
            },
            {
                title: "用户名",
                field: 'id',
                align: 'center',
                valign: 'middle'
            },
            {
                title: '性别',
                field: 'sex',
                align: 'center',
                valign: 'middle'
            },
            {
                title: '年龄',
                field: 'date',
                align: 'center',
                valign: 'middle'
            },
            {
                title: '部门',
                field: 'person',
                align: 'center'
            },
            {
                title: '职位',
                field: 'oper',
                align: 'center',
                valign: 'middle'
            },
            {
                title: '角色',
                field: 'title',
                align: 'center',
                valign: 'middle'
            },
            {
                title: '备注',
                field: 'person',
                align: 'center'
            }

        ]
    });
}
$(function(){
    table()
})