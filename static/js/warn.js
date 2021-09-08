/**
 * Created by Administrator on 2018/9/28.
 */
$(function(){
    list();
    window.parent.Right()
});
function warnType(type){
    if(type == 1){
        return '<div class="warn-btn blink " style="color: red;">【待处理】</div>';
    }else if(type == 2){
        return '<div class="warn-btn">【处理中】</div>';
    }else{
        return '<div class="warn-btn " style="color: green">【已处理】</div>'
    }
}
function warnState(state){
    if(state == 0){
        return '';
    }else if(state == 1){
        return '<div class="urgent">xxx</div>';
    }
}
function list(){
    $.ajax({
        type: "get",
        url:'../json/warn.json',
        dataType: "json",
        success:function(result){
            var html='',type = "",state="";
            var data = result.data;
            for(var i =0;i<data.length;i++){
                type = warnType(data[i].type);
                state = warnState(data[i].state);
                html += '<li onclick="parent.aniPoint()"><div class="warn-item"><div class="warn-item-content">'+
                        '<div class="item-header">'+
                        '<div class="number">'+data[i].number+'</div>'+
                        '<div class="text">'+data[i].title+'</div>'+type+
                        '</div><div class="warn-middle"><div class="middle-left">名称：</div>'+
                        '<div class="middle-right">'+data[i].dic+'</div></div>'+
                        '<div class="warn-middle"><div class="middle-left">时间：</div>'+
                        '<div class="middle-right">'+data[i].time+'</div></div>'+
                        '<div class="warn-middle"><div class="middle-left">地点：</div>'+
                        '<div class="middle-right">'+data[i].address+'</div></div>'+state+'</div>'+
                        '<div class="item-bottom-main"><div class="item-bottom">'
            }
            $("#warn-list").html(html);
            warnList();
        }
    })
}
