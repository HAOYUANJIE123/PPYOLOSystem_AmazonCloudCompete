/**
 * Created by Administrator on 2018/9/28.
 */
$(function(){
    list();
    window.parent.Right()
});
function warnType(type){
    if(type == 1){
        return '<div class="warn-btn green">集结中</div>';

    }else{
        return '<div class="warn-btn finish">已结束</div>'
    }
}
function safeClass(cls){
    if(cls=="一级"){
        return '<div style="color: yellow">'+cls+'</div>';
    }
    else if(cls=="二级"){
        return  '<div style="color: orangered">'+cls+'</div>';
    }
    else{
        return  '<div style="color: red">'+cls+'</div>';
    }

}
//function warnState(state){
//    if(state == 0){
//        return '';
//    }else if(state == 1){
//        return '<div class="urgent"><img src="../images/urgent.png"></div>';
//    }
//}
function list(){
    $.ajax({
        type: "get",
        url:'../json/team.json',
        dataType: "json",
        success:function(result){
            var html='',type = "",cls="";
            var data = result.data;
            for(var i =0;i<data.length;i++){
                type = warnType(data[i].type);
                cls = safeClass(data[i].title);
                html += '<li><div class="warn-item"><div class="warn-item-content">'+
                        '<div class="item-header">'+
                        '<div class="number">'+data[i].number+'</div>'+
                        '<div class="text"></div>'+type+
                        '</div><div class="warn-middle"><div class="middle-left">名称：</div>'+
                        '<div class="middle-right">'+data[i].dic+'</div></div>'+
                        '<div class="warn-middle"><div class="middle-left">地址：</div>'+
                        '<div class="middle-right">'+data[i].address+'</div></div>'+
                        '<div class="warn-middle"><div class="middle-left">时间：</div>'+
                        '<div class="middle-right">'+data[i].time+'</div></div></div>'+
                        '<div class="item-bottom-main"><div class="item-bottom">'+
                        '<div><span onclick="parent.addLocal1()">集结地址</span></div>'+'<div><span >取消集结</span></div>'+
                    '<div><span onclick="parent.safeLine()"><a href="../pages/TeamBulidResSumit.html" target="myFrame" style="color: #a4ccef">信息查看</a></span></div>'+
                       '</div></div></div></li>'
            }
            $("#warn-list").html(html);
            warnList();
        }
    })
}
//function look(){
//
//     window.parent.document.getElementById("safe").setAttribute("href","safeActive.html");
//}