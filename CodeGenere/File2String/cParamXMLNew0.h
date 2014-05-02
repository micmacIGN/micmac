#ifndef Define_ParamXMLNew0
#define Define_ParamXMLNew0
#include "StdAfx.h"
// NOMORE ...
typedef enum
{
  eTestDump_0,
  eTestDump_1
} eTestDump;
void xml_init(eTestDump & aVal,cElXMLTree * aTree);
std::string  eToString(const eTestDump & aVal);

eTestDump  Str2eTestDump(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTestDump & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTestDump &);

void  BinaryUnDumpFromFile(eTestDump &,ELISE_fp &);

class cTD2REF
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTD2REF & anObj,cElXMLTree * aTree);


        std::string & S();
        const std::string & S()const ;

        std::list< int > & V();
        const std::list< int > & V()const ;
    private:
        std::string mS;
        std::list< int > mV;
};
cElXMLTree * ToXMLTree(const cTD2REF &);

void  BinaryDumpInFile(ELISE_fp &,const cTD2REF &);

void  BinaryUnDumpFromFile(cTD2REF &,ELISE_fp &);

/******************************************************/
/******************************************************/
/******************************************************/
class cCompos
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cCompos & anObj,cElXMLTree * aTree);


        double & A();
        const double & A()const ;

        Pt2dr & B();
        const Pt2dr & B()const ;
    private:
        double mA;
        Pt2dr mB;
};
cElXMLTree * ToXMLTree(const cCompos &);

void  BinaryDumpInFile(ELISE_fp &,const cCompos &);

void  BinaryUnDumpFromFile(cCompos &,ELISE_fp &);

/******************************************************/
/******************************************************/
/******************************************************/
class cTestDump
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTestDump & anObj,cElXMLTree * aTree);


        int & I();
        const int & I()const ;

        cTplValGesInit< Pt2dr > & D();
        const cTplValGesInit< Pt2dr > & D()const ;

        eTestDump & E();
        const eTestDump & E()const ;

        std::list< eTestDump > & V();
        const std::list< eTestDump > & V()const ;

        cTD2REF & R1();
        const cTD2REF & R1()const ;

        cTplValGesInit< cTD2REF > & R2();
        const cTplValGesInit< cTD2REF > & R2()const ;

        double & A();
        const double & A()const ;

        Pt2dr & B();
        const Pt2dr & B()const ;

        cCompos & Compos();
        const cCompos & Compos()const ;
    private:
        int mI;
        cTplValGesInit< Pt2dr > mD;
        eTestDump mE;
        std::list< eTestDump > mV;
        cTD2REF mR1;
        cTplValGesInit< cTD2REF > mR2;
        cCompos mCompos;
};
cElXMLTree * ToXMLTree(const cTestDump &);

void  BinaryDumpInFile(ELISE_fp &,const cTestDump &);

void  BinaryUnDumpFromFile(cTestDump &,ELISE_fp &);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_ParamXMLNew0
