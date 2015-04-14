#ifndef Define_ParamXMLNew0
#define Define_ParamXMLNew0
#include "StdAfx.h"
// NOMORE ...
typedef enum
{
  eTestDump_0,
  eTestDump_1,
  eTestDump_3
} eTestDump;
void xml_init(eTestDump & aVal,cElXMLTree * aTree);
std::string  eToString(const eTestDump & aVal);

eTestDump  Str2eTestDump(const std::string & aName);

cElXMLTree * ToXMLTree(const std::string & aNameTag,const eTestDump & anObj);

void  BinaryDumpInFile(ELISE_fp &,const eTestDump &);

std::string  Mangling( eTestDump *);

void  BinaryUnDumpFromFile(eTestDump &,ELISE_fp &);

class cTD2REF
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTD2REF & anObj,cElXMLTree * aTree);


        std::string & K();
        const std::string & K()const ;

        std::list< int > & V();
        const std::list< int > & V()const ;
    private:
        std::string mK;
        std::list< int > mV;
};
cElXMLTree * ToXMLTree(const cTD2REF &);

void  BinaryDumpInFile(ELISE_fp &,const cTD2REF &);

void  BinaryUnDumpFromFile(cTD2REF &,ELISE_fp &);

std::string  Mangling( cTD2REF *);

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

std::string  Mangling( cCompos *);

/******************************************************/
/******************************************************/
/******************************************************/
class cTestDump
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTestDump & anObj,cElXMLTree * aTree);


        cTplValGesInit< int > & I();
        const cTplValGesInit< int > & I()const ;

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

        std::list< cTD2REF > & R3();
        const std::list< cTD2REF > & R3()const ;

        std::vector< cTD2REF > & R4();
        const std::vector< cTD2REF > & R4()const ;

        double & A();
        const double & A()const ;

        Pt2dr & B();
        const Pt2dr & B()const ;

        cCompos & Compos();
        const cCompos & Compos()const ;
    private:
        cTplValGesInit< int > mI;
        cTplValGesInit< Pt2dr > mD;
        eTestDump mE;
        std::list< eTestDump > mV;
        cTD2REF mR1;
        cTplValGesInit< cTD2REF > mR2;
        std::list< cTD2REF > mR3;
        std::vector< cTD2REF > mR4;
        cCompos mCompos;
};
cElXMLTree * ToXMLTree(const cTestDump &);

void  BinaryDumpInFile(ELISE_fp &,const cTestDump &);

void  BinaryUnDumpFromFile(cTestDump &,ELISE_fp &);

std::string  Mangling( cTestDump *);

/******************************************************/
/******************************************************/
/******************************************************/
class cR5
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cR5 & anObj,cElXMLTree * aTree);


        std::string & IdImage();
        const std::string & IdImage()const ;
    private:
        std::string mIdImage;
};
cElXMLTree * ToXMLTree(const cR5 &);

void  BinaryDumpInFile(ELISE_fp &,const cR5 &);

void  BinaryUnDumpFromFile(cR5 &,ELISE_fp &);

std::string  Mangling( cR5 *);

/******************************************************/
/******************************************************/
/******************************************************/
class cTestNoDump
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTestNoDump & anObj,cElXMLTree * aTree);


        std::map< std::string,cR5 > & R5();
        const std::map< std::string,cR5 > & R5()const ;

        int & AA();
        const int & AA()const ;

        std::vector<int> & vvAA();
        const std::vector<int> & vvAA()const ;
    private:
        std::map< std::string,cR5 > mR5;
        int mAA;
        std::vector<int> mvvAA;
};
cElXMLTree * ToXMLTree(const cTestNoDump &);

void  BinaryDumpInFile(ELISE_fp &,const cTestNoDump &);

void  BinaryUnDumpFromFile(cTestNoDump &,ELISE_fp &);

std::string  Mangling( cTestNoDump *);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_ParamXMLNew0
