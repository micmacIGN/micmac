//extracted from XML_GEN/SuperposImage.h
#ifndef SUPERPOSIMAGE_EXTRACT_H
#define SUPERPOSIMAGE_EXTRACT_H

class cXml_TopoTriplet
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_TopoTriplet & anObj,cElXMLTree * aTree);


        std::list< cXml_OneTriplet > & Triplets();
        const std::list< cXml_OneTriplet > & Triplets()const ;
    private:
        std::list< cXml_OneTriplet > mTriplets;
};
class cXml_OneTriplet
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_OneTriplet & anObj,cElXMLTree * aTree);


        std::string  Name1();
        //const std::string & Name1()const ;

        std::string Name2();
        //const std::string & Name2()const ;

        std::string Name3();
        //const std::string & Name3()const ;
    private:
        std::string mName1;
        std::string mName2;
        std::string mName3;
};

class cXml_Ori3ImInit
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_Ori3ImInit & anObj,cElXMLTree * aTree);


        cXml_Rotation & Ori2On1();
        const cXml_Rotation & Ori2On1()const ;

        cXml_Rotation & Ori3On1();
        const cXml_Rotation & Ori3On1()const ;

        int  NbTriplet();
        //const int & NbTriplet()const ;

        double & ResiduTriplet();
        const double & ResiduTriplet()const ;

        double  BSurH();
        //const double & BSurH()const ;

        Pt3dr & PMed();
        const Pt3dr & PMed()const ;

        cXml_Elips3D & Elips();
        const cXml_Elips3D & Elips()const ;
    private:
        cXml_Rotation mOri2On1;
        cXml_Rotation mOri3On1;
        int mNbTriplet;
        double mResiduTriplet;
        double mBSurH;
        Pt3dr mPMed;
        cXml_Elips3D mElips;
};
class cXml_Rotation
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cXml_Rotation & anObj,cElXMLTree * aTree);


        cTypeCodageMatr  Ori();
        //const cTypeCodageMatr & Ori()const ;

        Pt3dr  Centre();
        //const Pt3dr & Centre()const ;
    private:
        cTypeCodageMatr mOri;
        Pt3dr mCentre;
};
class cTypeCodageMatr
{
    public:
        cGlobXmlGen mGXml;

        friend void xml_init(cTypeCodageMatr & anObj,cElXMLTree * aTree);


        Pt3dr  L1();
        //const Pt3dr & L1()const ;

        Pt3dr  L2();
        //const Pt3dr & L2()const ;

        Pt3dr  L3();
        //const Pt3dr & L3()const ;

        cTplValGesInit< bool > & TrueRot();
        const cTplValGesInit< bool > & TrueRot()const ;
    private:
        Pt3dr mL1;
        Pt3dr mL2;
        Pt3dr mL3;
        cTplValGesInit< bool > mTrueRot;
};

#endif //SUPERPOSIMAGE_EXTRACT_H
