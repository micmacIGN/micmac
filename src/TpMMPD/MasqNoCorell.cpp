#include <vector>
#include <string>
#include "StdAfx.h"

int MasqNoCorell_main(int argc, char **argv) {

    bool verbose = false;

    std::string aFullMNSPattern;    // pattern des MNS corelles
    std::string aFullMasqPattern;   // pattern des Masques de correllation
    std::string aNameResult;// un fichier resultat
    int valDef = 0;

    ElInitArgMain
    (
        argc, argv,
        LArgMain()  << EAMC(aFullMNSPattern, "Pattern of corresponding input MNS", eSAM_IsPatFile)
                    << EAMC(aFullMasqPattern, "Pattern of corresponding input Masq", eSAM_IsPatFile)
                    << EAMC(aNameResult," output ortho filename"),
        LArgMain()  << EAM(valDef," value for NO-DATA")

     );
    std::cout << "Input MNS  : "<<aFullMNSPattern<<std::endl;
    std::cout << "Input Masq : "<<aFullMasqPattern<<std::endl;
    std::cout << "Output MNS : "<<aNameResult<<std::endl;

    std::string aDirMNS,aPatMNS;
    SplitDirAndFile(aDirMNS,aPatMNS,aFullMNSPattern);
    std::cout << "MNS dir : "<<aDirMNS<<std::endl;
    std::cout << "MNS pattern : "<<aPatMNS<<std::endl;
    std::string aDirMasq,aPatMasq;
    SplitDirAndFile(aDirMasq,aPatMasq,aFullMasqPattern);
    std::cout << "Masq dir : "<<aDirMasq<<std::endl;
    std::cout << "Masq pattern : "<<aPatMasq<<std::endl;

    // Chargement des MNS
    cInterfChantierNameManipulateur * aICNM_MNS=cInterfChantierNameManipulateur::BasicAlloc(aDirMNS);
    std::vector<std::string> aSetMNS = *(aICNM_MNS->Get(aPatMNS));
    std::vector<const TIm2D<REAL4,REAL8> *> aVMNS;
    for(size_t i=0;i<aSetMNS.size();++i)
    {
        std::string nom = aDirMNS+aSetMNS[i];
        std::cout << "MNS "<<i<<" : "<<nom<<std::endl;
        std::cout << "chargement ..."<<std::endl;
        aVMNS.push_back(new TIm2D<REAL4,REAL8>(Im2D<REAL4,REAL8>::FromFileStd(nom)));
        std::cout << "ok"<<std::endl;
    }

    // Chargement des Masques
    cInterfChantierNameManipulateur * aICNM_Masq=cInterfChantierNameManipulateur::BasicAlloc(aDirMasq);
    std::vector<std::string> aSetMasq = *(aICNM_Masq->Get(aPatMasq));
    std::vector<const TIm2D<U_INT1,INT4> *> aVMasq;
    for(size_t i=0;i<aSetMasq.size();++i)
    {
        std::string nom = aDirMasq+aSetMasq[i];
        std::cout << "Masq "<<i<<" : "<<nom<<std::endl;
        std::cout << "chargement ..."<<std::endl;
        aVMasq.push_back(new TIm2D<U_INT1,INT4>(Im2D<U_INT1,INT4>::FromFileStd(nom)));
        std::cout << "ok"<<std::endl;
    }

    // Creation de l'ortho
    int NC = aVMNS[0]->_the_im.tx();
    int NL = aVMNS[0]->_the_im.ty();
    Pt2di SzOrtho(NC,NL);
    TIm2D<REAL4,REAL8>* ptrOrtho = new TIm2D<REAL4,REAL8>(SzOrtho);

    for(int l=0;l<NL;++l)
    {
        for(int c=0;c<NC;++c)
        {
            Pt2di ptI;
            ptI.x = c;
            ptI.y = l;

            int valMasque = aVMasq[0]->get(ptI);
            if (valMasque)
            {
                double radio = aVMNS[0]->get(ptI);
                if (radio > -100000)
                {
                    //std::cout << "MNS ori " << c << " " << l << " : " << radio << std::endl;
                    ptrOrtho->oset(ptI,radio);
                    //double radioOUT = ptrOrtho->get(ptI);
                    //std::cout << "Result  " << c << " " << l << " : " << radioOUT << std::endl;
                }
            }
            else
                ptrOrtho->oset(ptI,valDef);
        }
    }


    Tiff_Im out(aNameResult.c_str(), ptrOrtho->sz(),GenIm::real4,Tiff_Im::No_Compr,Tiff_Im::BlackIsZero);
        ELISE_COPY(ptrOrtho->_the_im.all_pts(),ptrOrtho->_the_im.in(),out.out());

    return 0;
}
