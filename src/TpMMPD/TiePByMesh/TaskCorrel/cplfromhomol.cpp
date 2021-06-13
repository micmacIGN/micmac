#include <stdio.h>
#include "StdAfx.h"
#include "../InitOutil.h"

/******************************************************************************
The main function.
******************************************************************************/


void MakeXMLPairImg(vector<CplString> & cplImg, string & outXML)
{
    ofstream XMLfile;
    XMLfile.open (outXML.c_str());
    XMLfile<<"<?xml version=\"1.0\" ?>\n";
    XMLfile<<"  <SauvegardeNamedRel>\n";
    for (uint aKC=0; aKC<cplImg.size(); aKC++)
        XMLfile<<"    <Cple>"<<cplImg[aKC].img1<<" "<<cplImg[aKC].img2<<"</Cple>\n";
    XMLfile<<"  </SauvegardeNamedRel>\n";
    XMLfile.close();
}

vector<CplString> fsnPairImg(vector<CplString> & cplImg)
{
    vector<CplString> res;
    for (uint i=0; i<cplImg.size(); i++)
    {
        CplString aCpl =  cplImg[i];
        CplString aCplI; aCplI.img1 = aCpl.img2; aCplI.img2 = aCpl.img1;
        bool found = false;
        for (uint k=0; k<res.size(); k++)
        {
            CplString aCplres = res[k];
            if (aCplI.img1 == aCplres.img1 && aCplI.img2 == aCplres.img2)
            {
                found = true;
                break;
            }
        }
        if (!found)
            res.push_back(aCpl);
    }
    return res;
}

int CplFromHomol_main(int argc,char ** argv)
{
    cout<<"******************************************************************"<<endl;
    cout<<"*    CplFromHomol - creat XML of pair image from Homol exist     *"<<endl;
    cout<<"******************************************************************"<<endl;

    string outXML = "PairHomol.xml";
    string aHomolIn = "Homol";
    bool fusion = false;

    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()
                << EAMC(aHomolIn, "Input Homol",  eSAM_IsDir),
                //optional arguments
                LArgMain()
                << EAM(outXML, "OutXML", true, "name XML file output (default = PairHomol.xml")
                << EAM(fusion, "fusion", true, "eliminate pair image inverse - default=false")
                );

    //if (MMVisualMode) return EXIT_SUCCESS;
    std::list<cElFilename> LPatis;


    ctPath * aPathHomol = new ctPath(aHomolIn);
    aPathHomol->getContent(LPatis);

    std::vector<cElFilename> VPatis;
    std::list<cElFilename>::iterator NPatis = LPatis.begin();
    while ( NPatis!=LPatis.end() )
        VPatis.push_back(*NPatis++);

    vector<CplString> VcplImg;
    cout<<"Nb Patis : "<<VPatis.size()<<endl;
    for (uint aKL=0; aKL<VPatis.size(); aKL++)
    {
        //creat ctPath for each Patis
        cElFilename aPatis = VPatis[aKL];
        //cout<<aPatis.m_basename<<endl;
        string NImg1P = aPatis.m_basename; //need to remove "Patis"
        std::string NImg1 = NImg1P.substr (6,NImg1P.length()-5);     // "think"
        //cout<<" +Img1: "<<NImg1<<endl;

        std::list<cElFilename> Cur_DatHomol;
        ctPath aPatisP(aPatis.m_path.str() + aPatis.m_basename);
        aPatisP.getContent(Cur_DatHomol);
        //cout<<" ++NbHomolDat: "<<Cur_DatHomol.size()<<endl<<endl;


        std::list<cElFilename>::iterator DatPath = Cur_DatHomol.begin();
        while ( DatPath!=Cur_DatHomol.end() )
        {
            string aNImg2D = (*DatPath++).m_basename; //need to remove ".dat"
            std::string NImg2 = aNImg2D.substr (0,aNImg2D.length()-4);     // "think"
            //cout<<" +Img2: "<<NImg2<<endl;
            CplString aCpl; aCpl.img1 = NImg1; aCpl.img2 = NImg2;
            VcplImg.push_back(aCpl);
        }
    }
    vector<CplString> aRes = VcplImg;
    if (fusion)
        aRes = fsnPairImg(VcplImg);
    MakeXMLPairImg(aRes, outXML);
    cout<<"Done! NbCpl: "<<aRes.size()<<endl;

    /*
    std::vector<cElFilename> LHomol;
    std::list<cElFilename>::iterator FNPath = LAllFile.begin();
    while ( FNPath!=LAllFile.end() )
    {
        cout<<(*FNPath++).m_path<<endl;
        cout<<(*FNPath++).m_basename<<endl;
        LHomol.push_back(*FNPath++);
    }

    std::list<ctPath> LAllPatis (LAllFile.size());

    cout<<"Found : "<<LAllPatis.size()<<endl;
    int itV =0;
    std::list<ctPath>::iterator FNPatis = LAllPatis.begin();
    while ( FNPatis!=LAllPatis.end() )
    {
        (FNPatis++) = new ctPath(LHomol[itV].m_basename + LHomol[itV].m_path);
        itV++;
    }
*/
    return EXIT_SUCCESS;
}
