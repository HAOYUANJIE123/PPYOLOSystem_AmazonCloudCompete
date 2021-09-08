/**
 * Created by Administrator on 2018/9/28.
 */
$(function(){
    list();

});

function list(){
    $.ajax({
        type: "get",
        url:'../json/teamres.json',
        dataType: "json",
        success:function(result){
            var html="";
            var data = result.data;
            for(var i =0;i<data.length;i++){

                html +=' <li><div class="res-list-left"><input type="checkbox" class="res-checkbox01"><span class="res-num-span">'+data[i].number+'</span></div><div class="res-list-right"><span class="left">'+data[i].title+'</span><span class="right">'+data[i].type+'</span></div></li>'
            }
            $("#res-list").html(html);

        }
    })
}
//function look(){
//
//     window.parent.document.getElementById("safe").setAttribute("href","safeActive.html");
//}
function  addTeam(){
    //layer.open({
    //    type: 2,
    //    title: '新增详情',
    //    shade: 0.5,
    //    skin: 'layui-layer-rim',
    //    area: ['60%', '60%'],
    //    shadeClose: true,
    //    closeBtn: 2,
    //    content: 'addTeam.html'
    //})

    parent.art.dialog.open('addTeam.html',
        {title: '详情信息', width: '1200px', height: '600px',shade: 0.5,mask:true});

}