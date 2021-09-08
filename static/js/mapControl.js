/**
 * Created by 30947 on 2018/9/28.
 */
//表格部分
$(function(){
    table();
    getData();
})
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
                title: "姓名",
                field: 'name',
                align: 'center',
                valign: 'middle'
            },
            {
                title: '在岗状态',
                field: 'type',
                align: 'center',
                valign: 'middle',
                formatter:function(val,row){
                   if(val=="不在岗"){
                       return '<div style="color: #cccccc">'+val+'</div>'
                   }
                    else{
                       return'<div style="color: #FFFFff">'+val+'</div>'
                   }
                }
            },
            {
                title: '时间段',
                field: 'time',
                align: 'center',
                valign: 'middle'
            }


        ]
    });


}
function getData(){
    $.ajax({
        type:"get",
        url:"../json/mapControl.json",
        success:function(data){
            var data=data;

                $("#per").text(data.data[0].per);
                $("#car").text(data.data[0].car);
                $("#video").text(data.data[0].video);
                $("#set").text(data.data[0].set);
                var num=parseInt(data.data[0].casetol)/10000;
                var num_sp=num.toFixed(4).split(".");
                var str=num_sp.join("");
                var newArr=str.split("");
               var html="";
            for(i=0;i<newArr.length;i++){

                if(newArr[i]==0){
                    html="<span class='color_write blink'>"+newArr[i]+"</span>";
                }
                else{
                    html="<span>"+newArr[i]+"</span>";
                }
                $("#tol").append(html);

            }








        }

    })
}
