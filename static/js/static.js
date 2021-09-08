/**
 * Created by 30947 on 2018/6/11.
 */
$(function(){
char1();
    char2();
    char3();
    char4();
    table();
    changeDiv();

})

function char1() {

    var myChart = echarts.init($("#chart01")[0]);
//app.title = '堆叠柱状图';


    option = {

        tooltip : {
            trigger: 'item',
            formatter: "{a} <br/>{b} : {c} ({d}%)"
        },
        legend: {
            x : 'center',
            y : 'top',
            data:['指标一','指标二','指标三','指标四','指标五']
        },

        calculable : false,
        series : [
            {
                name:'指标一',
                type:'pie',
                radius : '50%',
                center: ['20%', '50%'],
                data:[
                    {value:335, name:'指标一'},
                    {value:310, name:'指标二'},
                    {value:234, name:'指标三'}

                ]
            },

            {
                name:'指标一',
                type:'pie',
                radius : '50%',
                center: ['70%', '50%'],
                data:[
                    {value:335, name:'指标四'},
                    {value:310, name:'指标五'}

                ]
            }
        ]
    };
    myChart.setOption(option);
    window.addEventListener('resize', function () {myChart.resize();})

}
function char2() {

    var myChart = echarts.init($("#chart02")[0]);
//app.title = '堆叠柱状图';


    option = {
        tooltip : {
            trigger: 'axis'
        },
        legend: {
            data:['指标一','指标二']
        },

        calculable : true,
        xAxis : [
            {
                type : 'category',
                boundaryGap : false,
                show:false,
                data : ['周一','周二','周三','周四','周五','周六','周日','周一','周二','周三','周四','周五','周六','周日']

            }
        ],
        yAxis : [
            {
                type : 'value'
            }
        ],
        series : [
            {
                name:'指标一',
                type:'line',
                stack: '总量',
                data:[120, 132, 101, 134, 90, 230, 210,120, 132, 101, 134, 90, 230, 210]
            },
            {
                name:'指标二',
                type:'line',
                stack: '总量',
                data:[220, 182, 191, 234, 290, 330, 310,120, 132, 101, 134, 90, 230, 210]
            }

        ]
    };
    myChart.setOption(option);
    window.addEventListener('resize', function () {myChart.resize();})

}
function char3() {

    var myChart = echarts.init($("#chart03")[0]);
//app.title = '堆叠柱状图';


    option = {

        tooltip : {
            trigger: 'axis'
        },
        legend: {
            data:['指标一','指标二']
        },

        calculable : true,
        xAxis : [
            {
                type : 'category',
                show:false,
                data : ['1月','2月','3月','4月','5月','6月','7月','8月','9月','10月','11月','12月']
            }
        ],
        yAxis : [
            {
                type : 'value'
            }
        ],
        series : [
            {
                name:'指标一',
                type:'bar',
                data:[2.0, 4.9, 7.0, 23.2, 25.6, 76.7, 135.6, 162.2, 32.6, 20.0, 6.4, 3.3],


            },
            {
                name:'指标二',
                type:'bar',
                data:[2.6, 5.9, 9.0, 26.4, 28.7, 70.7, 175.6, 182.2, 48.7, 18.8, 6.0, 2.3],


            }
        ]
    };

    myChart.setOption(option);
    window.addEventListener('resize', function () {myChart.resize();})

}
function char4() {

    var myChart = echarts.init($("#chart04")[0]);
//app.title = '堆叠柱状图';


    option = {

        tooltip : {
            trigger: 'axis'
        },
        legend: {
            data:['指标一','指标二']
        },

        calculable : true,
        xAxis : [
            {
                type : 'category',
                boundaryGap : false,
                show:false,
                data : ['周一','周二','周三','周四','周五','周六','周日']
            }
        ],
        yAxis : [
            {
                type : 'value'
            }
        ],
        series : [
            {
                name:'指标一',
                type:'line',
                smooth:true,
                itemStyle: {normal: {areaStyle: {type: 'default'}}},
                data:[10, 12, 21, 54, 260, 830, 710]
            },
            {
                name:'指标二',
                type:'line',
                smooth:true,
                itemStyle: {normal: {areaStyle: {type: 'default'}}},
                data:[30, 182, 434, 791, 390, 30, 10]
            }

        ]
    };


    myChart.setOption(option);
    window.addEventListener('resize', function () {myChart.resize();})

}
function table(){
    $('#table').bootstrapTable({
        method: "get",
        url: "../json/static.json",
        striped: true,
        singleSelect: false,
        dataType: "json",
        pagination: true, //分页
        pageSize: 10,
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
                title: '编号',
                field: 'no',
                align: 'center',
                valign: 'middle'
            }
            ,

            {
                title: '总数',
                field: 'num',
                align: 'center',
                valign: 'middle'
            }
            ,

            {
                title: '指标一',
                field: 'num01',
                align: 'center',
                valign: 'middle'
            }
            ,

            {
                title: '指标二',
                field: 'num02',
                align: 'center',
                valign: 'middle'
            }
            ,

            {
                title: '指标三',
                field: 'num03',
                align: 'center',
                valign: 'middle'
            }
            ,

            {
                title: '指标四',
                field: 'lv',
                align: 'center',
                valign: 'middle'
            }
            ,

            {
                title: '指标五',
                field: 'lv01',
                align: 'center',
                valign: 'middle'
            }
            ,

            {
                title: '指标六',
                field: 'lv02',
                align: 'center',
                valign: 'middle'
            }
        ]
    });


}
function changeDiv(){
    $(".static_tab a").click(function(){
        var ins=$(this).index();
        $(".static_con .static_con_div").eq(ins).show().siblings(".static_con_div").hide();
    })
}