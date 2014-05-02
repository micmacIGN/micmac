#ifndef Define_ParamXMLNew0
#define Define_ParamXMLNew0
#include "StdAfx.h"
// NOMORE ...
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

        double & A();
        const double & A()const ;

        Pt2dr & B();
        const Pt2dr & B()const ;

        cCompos & Compos();
        const cCompos & Compos()const ;
    private:
        int mI;
        cCompos mCompos;
};
cElXMLTree * ToXMLTree(const cTestDump &);

/******************************************************/
/******************************************************/
/******************************************************/
// };
#endif // Define_ParamXMLNew0
