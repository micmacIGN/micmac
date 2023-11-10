import sys
import os
import json
import subprocess
import re

debug=0

cur_word=sys.argv[1]
comp_cword=int(sys.argv[2])
comp_line=os.getenv('COMP_LINE')
screen_columns=int(os.getenv('COLUMNS','80'))


if debug>1:
  print ('\n',file=sys.stderr)
  print (f'1 curword=<{cur_word}>',file=sys.stderr)
  print (f'2 cword=<{comp_cword}>',file=sys.stderr)
  print (f'COMP_LINE=<{comp_line}>',file=sys.stderr)


def parseList(s) -> list:
    if not s:
        return [None,None]
    return [int(x) if x!='' else None for x in s[1:-1].split(',')]

def print_line(s) -> None:
    if not s:
        return
    spaces_needed = screen_columns - len(s) % screen_columns - 1
    print (s + ' ' * spaces_needed)
    print(' ')

def printMsgExit(s,code=0) -> None:
    print_line(s)
    print('Options:-o nosort')
    sys.exit(code)

    

def printFilter(word,words,option='') -> None:
    word_lower = word.lower()
    matches=[w for w in sorted(words) if w.lower().startswith(word_lower)]
    if len(matches) == 1:
        print(re.sub(r'^=','',matches[0]))
    else:
        for w in matches:
            print(re.sub(r'^=','',word + w[len(word):]))
    print (f"Options:-o nosort {option}")

def printSpecHelp(all_specs, spec, value) -> None:
    atype = spec['type']
    msg_type = atype.replace('std::','')
    semantic = spec.get('semantic')
    allowed = spec.get('allowed')
    vrange = parseList(spec.get('range'))
    vsize = parseList(spec.get('vsize'))
    vector = re.search(r'vector<(.*)>',msg_type)
    if vector:
        msg_type='[' + vector.group(1) + ',...]'
    elif atype == 'bool':
        allowed=['True','1','False','0']

    msg=None
    msg = f'>Expect <{msg_type}>: {spec["comment"]}'
    if allowed:
        msg += ' (Value one of [' + ','.join(allowed) + '])'
    if semantic and debug>0:
        msg += ' SEMANTIC:{' + ','.join(semantic) + '}'

    if vector and vsize[0] and vsize[0]>1:
        if not value:
            printFilter (value,'[','-o nospace')
            return
        printMsgExit(msg)

    if not value:
        print_line(msg)

    if vrange[0] and vrange[1]:
        if vrange[1] - vrange[0] < 20:
            printFilter(value,[str(i) for i in range(vrange[0],vrange[1]+1)])
        return
        
        
    if allowed:
        printFilter(value,allowed)
        return
        
    if msg_type != 'string':
        printMsgExit('')
    if not semantic:
        printMsgExit('')

    dir_types = all_specs["config"]["eTa2007DirTypes"]
    dir_type = set(semantic).intersection(dir_types)
    if len(dir_type) == 1:
        dir_type = list(dir_type)[0]
        sub_dir = all_specs['config']['MMVIIDirPhp'] +  dir_type
        print(f"File:pushd {sub_dir} >/dev/null 2>&1 && compgen -d -- '{value}'  && popd  >/dev/null 2>&1")
        return

    file_types = all_specs["config"]["eTa2007FileTypes"]
    file_types += ('MFP','FDP','I','Out','OptEx')
    file_type = set(semantic).intersection(file_types)
    if len(file_type) > 0:
        print(f"File:compgen -f -- '{value}'")
    if "DP" in semantic:
        print(f"File:compgen -d -- '{value}'")

def getAllSpecs() -> dict:
    try:
        result=subprocess.run(['MMVII','GenArgsSpec'],stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True)
    except:
        printMsgExit('>ERROR: MMVII not found.',1)
    try:
        all_specs=json.loads(result.stdout)
    except:
        printMsgExit(">ERROR: Can't get args specification from MMVII.",1)
    return all_specs

def commandNames(applets) -> None:
    app_names=(a['name'] for a in applets)
    printFilter(cur_word,app_names)

def main() -> int:
    all_specs=getAllSpecs()
    try:
        applets=all_specs['applets']
    except:
        printMsgExit(">ERROR: Can't get args specification from MMVII.",1)
    
    # 1st argument: MMVII command name
    if comp_cword == 1:
        commandNames(applets)
        return 0

    # else get applet specification
    command=comp_line.split()[1].lower()
    applet=[a for a in applets if a['name'].lower() == command]
    if len(applet) != 1:
        sys.exit(0)
    applet=applet[0]

    # Nth first arguments are mandatory
    if comp_cword < len(applet['mandatory'])+2:
        spec = applet['mandatory'][comp_cword-2]
        printSpecHelp(all_specs, spec, cur_word)
        return 0

    # Others are optionals
    arg_split=re.search(r'([-+a-zA-Z0-9_.]+)(=(.*))?',cur_word)
    #  no '='
    if arg_split==None or arg_split.start(2) < 0:
        normal=sorted((a['name']+'=' for a in applet['optional'] if a['level'] == 'normal'),key=str.lower)
        tuning=sorted((a['name']+'=' for a in applet['optional'] if a['level'] == 'tuning'),key=str.lower)
        common=sorted((a['name']+'=' for a in applet['optional'] if a['level'] == 'global'),key=str.lower)
        printFilter(cur_word,normal+tuning+common,'-o nospace')
        return 0
    # got '='
    arg=arg_split.group(1)
    specs = [ s for s in applet.get('optional') if s['name'].lower() == arg.lower() ] 
    if len(specs) != 1:
        return 0
    printSpecHelp(all_specs, specs[0], arg_split.group(3))
    return 0


if __name__ == '__main__':
    sys.exit(main())
