#include "../InitOutil.h"



/*
======================================================================================================
#                          <----Partie Couvert entre different series--->                            #
# ---->section Pre---------008-1417-----------1437-------------------1472                            #
#                          010-1536-----------1554-------------------1586--------section Suiv--->    #
#                          009-1473___________1493                                                   #
#                                             |   <- Section du regard                               #
#                                             |                                                      #
======================================================================================================
*/


string aPatSPre;
string aPatSSuiv;
string aPatSRega;
string outXML = "PairDiffSerie.xml";
bool withRega=0;
string aTurnPre, aTurnSuiv, aTurnRega;
int Line=0;

int XMLDiffSeries_main(int argc,char ** argv)
{
    ElInitArgMain
            (
                argc,argv,
                //mandatory arguments
                LArgMain()  << EAMC(aPatSPre, "Pattern of images in series Precedent",  eSAM_IsPatFile)
                            << EAMC(aPatSSuiv, "Pattern of images in series suivant",  eSAM_IsPatFile),
                //optional arguments
                LArgMain()
                << EAM(Line, "Line", true, "N° image adjacent to look for")
                << EAM(outXML, "out", true, "Fichier XML Out (def=PairDiffSerie.xml)")
                );

    if (MMVisualMode) return EXIT_SUCCESS;

    string aDirImages, aPatIm;
    SplitDirAndFile(aDirImages, aPatIm, aPatSPre);
    cInterfChantierNameManipulateur * aICNMPre = cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    vector<string>  vImgPre = *(aICNMPre->Get(aPatIm));

    cout<< "Series Pre : "<<vImgPre.size()<<endl;

    SplitDirAndFile(aDirImages, aPatIm, aPatSSuiv);
    cInterfChantierNameManipulateur * aICNMSuiv = cInterfChantierNameManipulateur::BasicAlloc(aDirImages);
    vector<string>  vImgSuiv = *(aICNMSuiv->Get(aPatIm));

    cout<< "Series Suiv : "<<vImgSuiv.size()<<endl;

    double ratio = vImgPre.size()/vImgSuiv.size();
    bool preAvant = 0;
    if (ratio < 1)
      ratio = 1/ratio;
    else
      preAvant = 1;
    vector<string>vImgMix;
    int indImgS = 0;
    int ratioR = ceil(ratio);
    int lineTapioca = 10;

    cout<<"Ratio : "<<ratioR<<" -PreAvant : "<<preAvant<<endl;

    if (!preAvant)
    {
        vector<string> temp;
        temp = vImgPre;
        vImgPre = vImgSuiv;
        vImgSuiv = temp;
    }


    for (int aKPre=0; aKPre<int(vImgPre.size()); aKPre++)
    {
        vImgMix.push_back(vImgPre[aKPre]);
        int addImgS = aKPre%ratioR;
        if (addImgS == 0)
        {
            if (indImgS < int(vImgSuiv.size()))
            {
                vImgMix.push_back(vImgSuiv[indImgS]);
                indImgS++;
            }
        }
    }
    if (indImgS < int(vImgSuiv.size()-1))
    {
        lineTapioca = ElMax(int(vImgSuiv.size()-1-indImgS), ElMax(lineTapioca, ratioR));
        for (int aKSuiv = indImgS+1; aKSuiv<int(vImgSuiv.size()); aKSuiv++)
            vImgMix.push_back(vImgSuiv[aKSuiv]);
    }

    cout<<"Get Char..."<<endl;
    getchar();
    for (int aKMix=0; aKMix<int(vImgMix.size()); aKMix++)
    {
        cout<<vImgMix[aKMix]<<endl;
    }
    //creer couple :
    if (EAMIsInit(&Line))
        lineTapioca=Line;
    cout<<"Creer Couple XML line: "<<lineTapioca<<endl;
    vector<CplString> vCplLine;
    for (int aKImg=0; aKImg<int(vImgMix.size()); aKImg++)
    {
        int indAvant = ElMax((aKImg-lineTapioca),0);
        int indApres = ElMin((aKImg+lineTapioca),int(vImgMix.size()-1));
        if (aKImg > 0)
        for (int aKAv=indAvant; aKAv<aKImg; aKAv++)
        {
            CplString aCpl;
            aCpl.img1 = vImgMix[aKImg];
            aCpl.img2 = vImgMix[aKAv];
            vCplLine.push_back(aCpl);
        }
        if (aKImg < int(vImgMix.size()-1))
        {
            for (int aKAp=aKImg+1; aKAp<indApres; aKAp++)
            {
                CplString aCpl;
                aCpl.img1 = vImgMix[aKImg];
                aCpl.img2 = vImgMix[aKAp];
                vCplLine.push_back(aCpl);
            }
        }

    }
    //fabriquer XML:
    vector<CplString> vCplFusion = fsnPairImg(vCplLine); //a voir si Tapioca File faire 1 ou 2 sens ?
    cout<<vCplLine.size()<<" cpls - "<<vCplFusion.size()<<" cpls fusion"<<endl;
    MakeXMLPairImg(vCplFusion, outXML);

    return EXIT_SUCCESS;
}
