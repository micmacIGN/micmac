function show_hide_div(state,object) {
//alert(object);
if(state) { object.style.display="block"; }
else { object.style.display="none"; }
}


function show_hide_div_id(state,object_id) {
if (!document.getElementById(object_id)) return false;
var object = document.getElementById(object_id);
if(state) { object.style.display="block"; }
else { object.style.display="none"; }
}


function show_hide_div_id_inline(state,object_id) {
var object = document.getElementById(object_id);
if(state) { object.style.display="inline"; }
else { object.style.display="none"; }
}

function check_show_hide_div(object_id) {
//alert(object_id);
var object = document.getElementById(object_id);
//alert(object);
var current_display = object.style.display;
if (current_display=='block') { object.style.display="none"; }
else { object.style.display="block"; }
}

function check_show_hide_div1(object_id) {
var object = document.getElementById(object_id);
var current_display = object.style.display;
if (current_display=='block') { object.style.display="none"; }
else { object.style.display="none"; }
}

function show_campaign(object_n) {
for (x=0;x<=5;x++)
{ var this_id = 'campaign_div_'+x;
  if (document.getElementById(this_id)) show_hide_div_id(0,this_id);
}
if (document.getElementById('campaign_div_'+object_n)) show_hide_div_id(1,'campaign_div_'+object_n);
}

function open_new_window(url,w,h,scrollbars) {
winLeft = (screen.width-800)/2; 
winTop = (screen.height-720)/2; 
new_window = window.open(url,'my_window','scrollbars='+scrollbars+',toolbar=0,menubar=0,resizable=0,dependent=0,status=0,width='+w+',height='+h+',left='+winLeft+',top='+winTop);
}

function go_to_delete(text,url) { if (confirm(text)) { location = url; } }

function checkAll(formId,cName,check) { for (i=0,n=formId.elements.length;i<n;i++) if (formId.elements[i].className.indexOf(cName) !=-1) formId.elements[i].checked = check; }

function preview_window(url)
{ newwindow=window.open(url,'preview_window','width='+(screen.availWidth-150)+',height='+(screen.availHeight-150)+',left=10,right=10,resizable=yes,scrollbars=yes,toolbar=yes,status=yes,location=yes');
  newwindow.focus();
}




function show_top_submenu(object_n) {
for (x=1;x<=50;x++)
{ var this_id = 'top_submenu_'+x;
  if (document.getElementById(this_id)) show_hide_div_id(0,this_id);
}
if (document.getElementById('top_submenu_'+object_n)) show_hide_div_id(1,'top_submenu_'+object_n);
}




function getObject(obj) {
var theObj;
if(document.all)
{ if(typeof obj=="string") { return document.all(obj); } 
  else { return obj.style; }
}
if(document.getElementById)
{ if(typeof obj=="string") { return document.getElementById(obj); }
  else { return obj.style; }
}
return null;
}
function Contar(field_id,text_id,texto,caracteres,line_id) {
var field_object = getObject(field_id);
var text_object = getObject(text_id);
update_text(field_id,line_id);
var longitud = caracteres - field_object.value.length;
if(longitud <= 0)
{ longitud=0;
  texto='<span class="disable"> '+texto+' </span>';
  field_object.value=field_object.value.substr(0,caracteres);
}
text_object.innerHTML = texto.replace("{CHAR}",longitud);
}
function update_text(field_id,line_id) {
if (line_id=='url1') return null;
var line_id1;
if ((field_id1 = document.getElementById(field_id)) && (line_id1 = document.getElementById(line_id)))
{ //alert(line_id+'---'+field_id1+'---'+line_id1);
  line_id1.innerHTML = document.getElementById(field_id).value;
}
return null;
}










function format_numbers(nStr) {
nStr += '';
x = nStr.split('.');
x1 = x[0];
x2 = x.length > 1 ? '.' + x[1] : '';
var rgx = /(\d+)(\d{3})/;
while (rgx.test(x1)) { x1 = x1.replace(rgx, '$1' + ',' + '$2'); }
return x1 + x2;
}
function write_new_value(text,id) {
if (x = document.getElementById(id)) { x.innerHTML = ''; x.innerHTML = text; }
}

function add_impressions() {
//var randomnumber = (Math.round((Math.random()*30)+1));
var randomnumber = 1;

var i_today = document.getElementById('i_today');
if(i_today.value=='NaN' || !i_today.value) { return false; }
var number = parseInt(i_today.value) + randomnumber;
var new_value = format_numbers(number);
write_new_value(new_value,'div_i_today');
i_today.value = number;

var i_total = document.getElementById('i_total');
if(i_total.value=='NaN' || !i_total.value) { return false; }
var number = parseInt(i_total.value) + randomnumber;
var new_value = format_numbers(number);
write_new_value(new_value,'div_i_total');
i_total.value = number;

var i_days = document.getElementById('i_days');
if(i_days.value=='NaN' || !i_days.value) { return false; }
var number = parseInt(i_days.value) + randomnumber;
var new_value = format_numbers(number);
write_new_value(new_value,'div_i_days');
i_days.value = number;

}
function add_clicks() {
var randomnumber = 1;

var c_today = document.getElementById('c_today');
if(c_today.value=='NaN' || !c_today.value) { return false; }
var number = parseInt(c_today.value) + randomnumber;
var new_value = format_numbers(number);
write_new_value(new_value,'div_c_today');
c_today.value = number;

var c_total = document.getElementById('c_total');
if(c_total.value=='NaN' || !c_total.value) { return false; }
var number = parseInt(c_total.value) + randomnumber;
var new_value = format_numbers(number);
write_new_value(new_value,'div_c_total');
c_total.value = number;

var c_days = document.getElementById('c_days');
if(c_days.value=='NaN' || !c_days.value) { return false; }
var number = parseInt(c_days.value) + randomnumber;
var new_value = format_numbers(number);
write_new_value(new_value,'div_c_days');
c_days.value = number;

}



/*###############################################################################*/

function show_waiting(object_id) {
var object = document.getElementById(object_id);
object.innerHTML = '<div><br><br><br><img src="'+site_url+'/images/waiting.gif" border="0"><br><br><br></div>';
}



function show_ajax_content(url,elementid) {
if (window.XMLHttpRequest)
{ // IE7+, Firefox, Chrome, Opera, Safari
  xmlhttp=new XMLHttpRequest();
}
else
{ // code for IE6, IE5
  xmlhttp=new ActiveXObject("Microsoft.XMLHTTP");
}
xmlhttp.onreadystatechange=function()
{ if (xmlhttp.readyState==4 && xmlhttp.status==200)
  { document.getElementById(elementid).innerHTML=xmlhttp.responseText; }
}
xmlhttp.open("GET",url,true);
xmlhttp.send();
}



function parse_ajax_request(form_id,php_script_url,element_id) {
var http_request = false;
//alert(form_id+'---'+php_script_url+'---'+element_id);
var request = '';
if (form_id)
{ for(i=0; i<form_id.elements.length; i++)
  { if (form_id.elements[i].type=='checkbox')
    { //alert(form_id.elements[i].checked);
	  if (form_id.elements[i].checked==true) request += form_id.elements[i].name+'=1&';
    }
    else if (form_id.elements[i].type=='radio')
    { if (form_id.elements[i].checked==true)
      { request += form_id.elements[i].name+"="+form_id.elements[i].value+'&';
      }
    }
    else
    { request += form_id.elements[i].name+"="+form_id.elements[i].value+'&';
      //alert("The field name is: " + form_id.elements[i].name + " and its value is: " + form_id.elements[i].value + ".<br />");
    }
  }
}
//alert(request);
//alert(form_id.string.value);
//var string = form_id.string.value;
//var request = "string="+string;
if (window.XMLHttpRequest) { http_request = new XMLHttpRequest(); } 
else if (window.ActiveXObject)
{ try { http_request = new ActiveXObject("Msxml2.XMLHTTP"); }
  catch (eror) { http_request = new ActiveXObject("Microsoft.XMLHTTP"); }
}
var redirect_url = '';
http_request.onreadystatechange = function() { show_ajax_result(http_request,element_id,redirect_url); };
http_request.open('GET',php_script_url+'?'+request,true);
http_request.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
http_request.send(request);
}


function process_ajax_form(form_id,php_script_url,element_id,redirect_url) {
var http_request = false;
//alert(form_id+'---'+php_script_url+'---'+element_id);
var request = '';
var form_element = document.getElementById(form_id);
if (form_element)
{ for(i=0; i<form_element.elements.length; i++)
  { if (form_element.elements[i].type=='checkbox')
    { //alert(form_element.elements[i].checked);
	  if (form_element.elements[i].checked==true) request += form_element.elements[i].name+'=1&';
    }
    else if (form_element.elements[i].type=='radio')
    { if (form_element.elements[i].checked==true)
      { request += form_element.elements[i].name+"="+form_element.elements[i].value+'&';
      }
    }
    else
    { request += form_element.elements[i].name+"="+form_element.elements[i].value+'&';
      //alert("The field name is: " + form_element.elements[i].name + " and its value is: " + form_element.elements[i].value + ".<br />");
    }
  }
}
if (window.XMLHttpRequest) { http_request = new XMLHttpRequest(); } 
else if (window.ActiveXObject)
{ try { http_request = new ActiveXObject("Msxml2.XMLHTTP"); }
  catch (eror) { http_request = new ActiveXObject("Microsoft.XMLHTTP"); }
}
http_request.onreadystatechange = function() { show_ajax_result(http_request,element_id,redirect_url); };
http_request.open('GET',php_script_url+'?'+request,true);
http_request.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
http_request.send(request);
}



function show_ajax_result(http_request,element_id,redirect_url) {
	//alert(http_request.readyState+'----'+element_id);
//  alert(redirect_url);

if (http_request.readyState == 4)
{ if (http_request.status == 200) {
  document.getElementById(element_id).innerHTML  = http_request.responseText;
  //alert(redirect_url);
  if (redirect_url) window.location = redirect_url;
  }
  else { alert('An error has occurred. Please try again.'); }
}
}

/*###############################################################################*/


function check_countries(campaign,checkit) {
//alert(campaign+'--'+checkit);
for (x=1;x<=500;x++)
{ var this_id = 'country_id_'+campaign+'_'+x;
  if (document.getElementById(this_id))
  { if (checkit==1) document.getElementById(this_id).checked = true;
    else document.getElementById(this_id).checked = false;
  }
}
if (document.getElementById('country_id_'+campaign+'_0')) { if (checkit) document.getElementById('country_id_'+campaign+'_0').checked = false; else  document.getElementById('country_id_'+campaign+'_0').checked = true; }
}


function check_all(field_name,campaign,checkit) {
//alert(field_name+'---'+campaign+'--'+checkit);
for (x=0;x<=500;x++)
{ var this_id = field_name+'_'+campaign+'_'+x;
  if (document.getElementById(this_id))
  { if (checkit==1) document.getElementById(this_id).checked = true;
    else document.getElementById(this_id).checked = false;
  }
}
}

/*###############################################################################*/


function checkbox_checked(checkbox_id) {
if (document.getElementById(checkbox_id).checked==true)
return true;
else
return false;
}


function add_city_hidden(element_id,checkbox_id,country,region,city,city_name) {
var hidden_element_id_a = element_id + 'hidden_a';
var new_checkbox_id = checkbox_id + 'a';
var checkbox = '<input type="checkbox" name="countries[]" value="'+country+'-'+region+'-'+city+'" checked id="'+new_checkbox_id+'">'+city_name+'<br>';
if (checkbox_checked(checkbox_id))
{ document.getElementById(element_id).style.display="block";
  if ((document.getElementById(hidden_element_id_a).innerHTML.indexOf(city + ','))==-1)
  { document.getElementById(element_id).innerHTML = document.getElementById(element_id).innerHTML + checkbox;
    document.getElementById(hidden_element_id_a).innerHTML = document.getElementById(hidden_element_id_a).innerHTML + ','+city+',';
  }
  else document.getElementById(new_checkbox_id).checked=true;
}
else
{ document.getElementById(new_checkbox_id).checked=false;
  //var str = document.getElementById(hidden_element_id_a).innerHTML;
  //document.getElementById(hidden_element_id_a).innerHTML = str.replace(','+city+',','');
}
}

/*###############################################################################*/

