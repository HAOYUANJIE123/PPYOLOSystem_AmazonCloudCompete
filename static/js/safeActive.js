/**
 * Created by 30947 on 2018/9/28.
 */
//表格部分
$(function(){

    getData();
    run();
    alam();
    table();
})
//安保活动
function getData(){
    $.ajax({
        type:"get",
        url:"../json/safeActive.json",
        success:function(res){
            var data=res.data;
            var name=data[0].name;
            var type=data[0].type;
            var cla=data[0].class;
            var time=data[0].time;
            var html="<div class='flex_con' ><label>活动名称：</label><span>"+name+"</span></div> " +
                "<div class='flex_con'><div class='flex_con_div'><label>活动类型：</label><span>"+type+"</span></div><div class='flex_con_div'><label>活动级别：</label><span class=‘color_blue’>"+cla+"</span></div></div>" +
                "<div class='flex_con'><label>活动日期：</label><span>"+time+"</span></div>";
            $("#one").append(html);


        }

    })
}
//行程安排
function run(){
    $.ajax({
        type:"get",
        url:"../json/safeActive.json",
        success:function(res){
            var data=res.data;

            $.each(data,function(index){
                var lab=data[index].lab;
                var labname=data[index].labname;
                var runtime=data[index].runtime;
                var runline=data[index].runtline;
               var html="<div class='flex_bg'><div class='flex_con' ><label>"+lab+"：</label><span>"+labname+"</span></div> <div class='flex_con'><label>行程时间：</label><span>"+runtime+"</span><a href='#' class='flex_a bg_blue'>查看</a> </div><div class='flex_con'><label>行程路线：</label><span>"+runline+"</span><a href='#' class='flex_a bg_sk'>定位</a> </div></div>";
                $("#two").append(html);

            })
        }

    })
}
//安保警力
function alam(){
    $.ajax({
        type:"get",
        url:"../json/alam.json",
        success:function(res){
            var data=res.data;
            var pernum=data[0].pernum;
            var videonum=data[0].videonum;
            var carnum=data[0].carnum;
            var ballnum=data[0].ballnum;
            var html="<div class='flex_con'><div class='flex_con_div'><label>警员：</label><span>"+pernum+"</span></div><div class='flex_con_div'><label>监控：</label><span class=‘color_blue’>"+videonum+"</span></div></div><div class='flex_con'><div class='flex_con_div'><label>警车：</label><span>"+carnum+"</span></div><div class='flex_con_div'><label>移动布控球：</label><span class=‘color_blue’>"+ballnum+"</span></div></div>";
            $("#three").append(html);
        }

    })
}
function table(){
    $('#table').bootstrapTable({
        method: "get",
        url: "../json/case.json",
        striped: true,
        singleSelect: false,
        dataType: "json",
        pagination: false, //分页
        pageSize: 3,
        pageNumber: 1,
        search: false, //显示搜索框
        contentType: "application/x-www-form-urlencoded",
        queryParams: null,
        //sidePagination: "server", //服务端请求
        columns: [

            {
                title: "排序",
                field: 'stor',
                align: 'center',
                valign: 'middle'
            },

            {
                title: '地址',
                field: 'addr',
                align: 'center',
                valign: 'middle'
            }


        ]
    });


}