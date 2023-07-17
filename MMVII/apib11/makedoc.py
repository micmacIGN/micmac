#!/usr/bin/env python3
"""
@author: cmeynard
"""

import xml.sax
import sys
import re
from collections import deque

class DoxygenHandler( xml.sax.ContentHandler ):
    def __init__(self):
        self.allDesc = dict()
        self.operators = {
            '<=': 'le', '>=': 'ge', '==': 'eq', '!=': 'ne', '[]': 'array',
            '+=': 'iadd', '-=': 'isub', '*=': 'imul', '/=': 'idiv',
            '%=': 'imod', '&=': 'iand', '|=': 'ior', '^=': 'ixor',
            '<<=': 'ilshift', '>>=': 'irshift', '++': 'inc', '--': 'dec',
            '<<': 'lshift', '>>': 'rshift',
            '&&': 'land', '||': 'lor','!': 'lnot',
            '~': 'bnot',  '&': 'band', '|': 'bor', '^': 'bxor',
            '+': 'add', '-': 'sub', '*': 'mul', '/': 'div', '%': 'mod',
            '<': 'lt', '>': 'gt', '=': 'assign', '()': 'call'
        }

    def cleanName(self,name):
        for symb,alias in self.operators.items():
            name = name.replace(f'operator{symb}', f'operator_{alias}')
        name = re.sub('<.*>', '', name)
        name = ''.join([ch if ch.isalnum() else '_' for ch in name])
        name = re.sub('_$', '', re.sub('_+', '_', name))
        return name

    def addDesc(self,name):
        if 'detail' in self.desc and self.desc['detail'] != '':
            desc = self.desc['detail']
        elif 'brief' in self.desc  and self.desc['brief'] != '':
            desc = self.desc['brief']
        else:
            desc=""
        desc = desc.strip()
        name = self.cleanName(name)
        if name in self.allDesc:
           self.allDesc[name].append(desc)
        else:
           self.allDesc[name] = [desc]

    def startDocument(self):
        self.inDefinition = False
        self.inClass = False
        self.inMember = False
        self.inPara = False
        self.element = None
        self.elements = deque()
        self.desc = dict()
        self.descType = None
        self.className = ''
        self.memberName = ''
        self.definition = ''
        self.indent = 0


    def compounddef_start(self, attributes):
        if attributes['kind'] in ('class','namespace'):
            self.inClass = True
            self.className = ''

    def compounddef_end(self):
        if self.inClass:
            self.addDesc(self.className)
        self.inClass = False
        self.className = ''


    def memberdef_start(self, attributes):
        if attributes['kind'] in ('function','variable') :
            self.inMember = True
            self.definition = ''
            self.memberName = ''

    def memberdef_end(self):
        if not self.inMember:
            return
        if '~' not in self.definition:
            self.addDesc(self.className + '_' + self.memberName)
        self.inMember = False
        self.definition = ''
        self.memberName = ''

    def definition_start(self, attributes):
        if not self.inMember:
            return
        self.inDefinition = True
        self.definition = ""

    def definition_end(self):
        if not self.inMember:
            return
        self.inDefinition = False


    def briefdescription_start(self, attributes):
        if not self.inClass and not self.inMember:
            return
        self.descType='brief'
        self.desc[self.descType] = ''

    def briefdescription_end(self):
        self.descType=None

    def detaileddescription_start(self, attributes):
        if not self.inClass and not self.inMember:
            return
        self.descType='detail'
        self.desc[self.descType] = ''

    def detaileddescription_end(self):
        self.descType=None

    def para_start(self, attributes):
        self.inPara = True

    def para_end(self):
        self.inPara = False
        if self.descType is None:
            return
        self.desc[self.descType] += '\n'

    def verbatiom_start(self, attributes):
        if self.descType is None:
            return
        self.desc[self.descType] += '\n'

    def itemizedlist_start(self, attributes):
        if self.descType is None:
            return
        self.indent += 4

    def itemizedlist_end(self):
        if self.descType is None:
            return
        self.indent -= 4

    def listitem_start(self, attributes):
        if self.descType is None:
            return
        self.desc[self.descType] += ' ' * self.indent + '- '

    def characters(self, content):
        if self.inPara and self.descType is not None:
            self.desc[self.descType] += content;
            return
        if self.inClass and self.element == 'compoundname':
            self.className += content;
            return
        if self.inMember and self.element == 'name':
            self.memberName += content;
            return
        if self.inDefinition:
            self.definition += content;
            return


    def startElement(self, tag, attributes):
        self.elements.append(self.element)
        self.element = tag
        func = getattr(DoxygenHandler,tag + '_start',None)
        if func:
            func(self,attributes)

    def endElement(self, tag):
        self.element = self.elements.pop()
        func = getattr(DoxygenHandler,tag + '_end',None)
        if func:
            func(self)


    def outputDesc(self):
        for name in sorted(self.allDesc):
            descs = self.allDesc[name]
            for count,desc in enumerate(descs):
                if count == 0:
                    print (f'static const char *__doc_{name} = R"doc({desc})doc";\n')
                else:
                    print (f'static const char *__doc_{name}_{count+1} = R"doc({desc})doc";\n')


def main():
    header = """/*
  This file contains docstrings for use in the Python bindings.
  Do not edit! They were automatically extracted by pybind11_mkdoc.
 */

#define __EXPAND(x)                                      x
#define __COUNT(_1, _2, _3, _4, _5, _6, _7, COUNT, ...)  COUNT
#define __VA_SIZE(...)                                   __EXPAND(__COUNT(__VA_ARGS__, 7, 6, 5, 4, 3, 2, 1))
#define __CAT1(a, b)                                     a ## b
#define __CAT2(a, b)                                     __CAT1(a, b)
#define __DOC1(n1)                                       __doc_##n1
#define __DOC2(n1, n2)                                   __doc_##n1##_##n2
#define __DOC3(n1, n2, n3)                               __doc_##n1##_##n2##_##n3
#define __DOC4(n1, n2, n3, n4)                           __doc_##n1##_##n2##_##n3##_##n4
#define __DOC5(n1, n2, n3, n4, n5)                       __doc_##n1##_##n2##_##n3##_##n4##_##n5
#define __DOC6(n1, n2, n3, n4, n5, n6)                   __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6
#define __DOC7(n1, n2, n3, n4, n5, n6, n7)               __doc_##n1##_##n2##_##n3##_##n4##_##n5##_##n6##_##n7
#define DOC(...)                                         __EXPAND(__EXPAND(__CAT2(__DOC, __VA_SIZE(__VA_ARGS__)))(__VA_ARGS__))

#if defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable"
#endif

"""
    footer = """
#if defined(__GNUG__)
#pragma GCC diagnostic pop
#endif
"""

    parser = xml.sax.make_parser()
    parser.setFeature(xml.sax.handler.feature_namespaces, 0)
    docHandler = DoxygenHandler()
    parser.setContentHandler( docHandler )
    for file in sys.argv[1:]:
        parser.parse(file)

    print(header);
    docHandler.outputDesc()
    print(footer)
    return 0

if __name__ == '__main__':
    sys.exit(main())
