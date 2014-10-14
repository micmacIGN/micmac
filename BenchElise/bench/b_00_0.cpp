/*eLiSe06/05/99
  
     Copyright (C) 1999 Marc PIERROT DESEILLIGNY

   eLiSe : Elements of a Linux Image Software Environment

This program is free software; you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation; either version 2 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program; if not, write to the Free Software
Foundation, Inc., 675 Mass Ave, Cambridge, MA 02139, USA.

  Author: Marc PIERROT DESEILLIGNY    IGN/MATIS  
Internet: Marc.Pierrot-Deseilligny@ign.fr
   Phone: (33) 01 43 98 81 28
eLiSe06/05/99*/



#include "StdAfx.h"
#include "bench.h"

class  BENCH_Flux_of_Byte :  public Flux_Of_Byte
{
    public :
          virtual int tell()
          {
              BENCH_ASSERT(0);
              return 1;
          }
};

class Natural_byte_flux : public BENCH_Flux_of_Byte
{
      public :

          Natural_byte_flux(): _cpt (0){}
          virtual U_INT1 Getc() {return _cpt++;}
          virtual void Putc(U_INT1) {}


          U_INT1 _cpt;
};

class Cste_byte_flux : public BENCH_Flux_of_Byte
{
      public :

          Cste_byte_flux(INT cste): _cste (cste){}
          virtual U_INT1 Getc() {return _cste;}
          virtual void Putc(U_INT1) {}

          U_INT1 _cste;
};

void bench_msb_varLI()
{
    {
         Natural_byte_flux NBF;
         MSBitFirst_Flux_Of_VarLI flval(&NBF,false);

         for (INT i =0 ; i < 256; i++)
             BENCH_ASSERT( flval.nexti(8) == i) ;
    }

    {
         Cste_byte_flux CBF(255);
         MSBitFirst_Flux_Of_VarLI flval(&CBF,false);

         for (INT i =0 ; i < 2000; i++)
         {
             INT nb = (i + i%7 + i%5 +i%3) % 30;
             BENCH_ASSERT( flval.nexti(nb) == ((1<< nb) -1)) ;
         }
    }

    {
         Cste_byte_flux CBF(15);
         MSBitFirst_Flux_Of_VarLI flval(&CBF,false);

         BENCH_ASSERT( flval.nexti(4) == 0);
         BENCH_ASSERT( flval.nexti(4) == 15);
         BENCH_ASSERT( flval.nexti(4) == 0);
         BENCH_ASSERT( flval.nexti(4) == 15);

         BENCH_ASSERT( flval.nexti(6) == 3);

         BENCH_ASSERT(flval.nexti(12) == (3 << 10) + (15 << 2));

         BENCH_ASSERT(flval.nexti(1) == 0);
         BENCH_ASSERT(flval.nexti(1) == 0);
         BENCH_ASSERT(flval.nexti(1) == 1);
         BENCH_ASSERT(flval.nexti(1) == 1);
         BENCH_ASSERT(flval.nexti(0) == 0);
         BENCH_ASSERT(flval.nexti(1) == 1);
         BENCH_ASSERT(flval.nexti(1) == 1);


    }
}

void bench_lsb_varLI()
{
    {
         Natural_byte_flux NBF;
         LSBitFirst_Flux_Of_VarLI flval(&NBF,false);

         for (INT i =0 ; i < 256; i++)
             BENCH_ASSERT( flval.nexti(8) == i) ;
    }

    {
         Cste_byte_flux CBF(255);
         LSBitFirst_Flux_Of_VarLI flval(&CBF,false);

         for (INT i =0 ; i < 2000; i++)
         {
             INT nb = (i + i%7 + i%5 +i%3) % 30;
             BENCH_ASSERT( flval.nexti(nb) == ((1<< nb) -1)) ;
         }
    }

    {
         Cste_byte_flux CBF(15);
         LSBitFirst_Flux_Of_VarLI flval(&CBF,false);

         BENCH_ASSERT( flval.nexti(4) == 15);
         BENCH_ASSERT( flval.nexti(4) == 0);
         BENCH_ASSERT( flval.nexti(4) == 15);
         BENCH_ASSERT( flval.nexti(4) == 0);

         BENCH_ASSERT( flval.nexti(3) == 7);
         BENCH_ASSERT( flval.nexti(6) == (1 << 5) +1);
         BENCH_ASSERT( flval.nexti(19) == 7 + (15<<7) + (15<<15));
    }

}


class Tab_Flx : public BENCH_Flux_of_Byte
{
      public :

          Tab_Flx(): _nb (0){for (int i=0;i<500;i++) _tab[i]=0;}

          void reset () {_nb = 0;}
          virtual U_INT1 Getc() {return _tab[_nb++];}
          virtual void Putc(U_INT1 v) {_tab[_nb++] = v;}


          INT _nb;
          U_INT1 _tab[500];
};

void bench_msb_out_varLI()
{
    Tab_Flx flx;
	INT i;

    MSBF_Flux_OutVarLI MSO(&flx,false);

    for ( i =0 ; i < 200; i++)
    {
        INT nbb = 1+i%3+i%5+i%7;
        INT val = (i+i*i+i%11) % (1<<nbb);
        MSO.puti(val,nbb);
    }
    MSO.reset();

    flx.reset();

    MSBitFirst_Flux_Of_VarLI   MSI(&flx,false);

    for ( i =0 ; i < 200; i++)
    {
        INT nbb = 1+i%3+i%5+i%7;
        INT val = (i+i*i+i%11) % (1<<nbb);
        BENCH_ASSERT(val==MSI.nexti(nbb)) ;
    }
}

void bench_lsb_out_varLI()
{

    Tab_Flx flx;
	INT i;

    LSBF_Flux_OutVarLI MSO(&flx,false);

    for ( i =0 ; i < 200; i++)
    {
        INT nbb = 1+i%3+i%5+i%7;
        INT val = (i+i*i+i%11) % (1<<nbb);
        MSO.puti(val,nbb);
    }
    MSO.reset();

    flx.reset();

    LSBitFirst_Flux_Of_VarLI   MSI(&flx,false);

    for ( i =0 ; i < 200; i++)
    {
        INT nbb = 1+i%3+i%5+i%7;
        INT val = (i+i*i+i%11) % (1<<nbb);
        BENCH_ASSERT(val==MSI.nexti(nbb)) ;
    }
}


/******************************************/

class FFOB : public Flux_Of_Byte
{
      public :

        int tell() {return _pile.nb();}
        U_INT1 Getc() {return _pile[_k++];}
        void Putc(U_INT1 c) {_pile.pushlast(c);}
        FFOB() : _k(0) {}
        // INT nb() {return _pile.nb();}

      private :

         INT            _k;
         ElFifo<U_INT1> _pile;
};                                       

class cTest_SAC
{
	public :
		cTest_SAC(INT aV0);
		void AddCode(INT aFreq,INT aCumul,INT aTot);
		~cTest_SAC();
	private :
          FFOB mFlx;
          MSBitFirst_Flux_Of_VarLI mMSI;
          MSBF_Flux_OutVarLI       mMSO;
          MS_RANGE_ENCODER         mEnc;
          MS_RANGE_DECODER         mDec;
	  std::vector<int>         mVFreq;
	  std::vector<int>         mVCumul;
	  std::vector<int>         mVTot;
	  INT                      mV0;
};


cTest_SAC::cTest_SAC(INT aV0) :
    mMSI   (&mFlx,false),
    mMSO   (&mFlx,false),
    mEnc   (&mMSO),
    mDec   (&mMSI),
    mV0    (aV0)
{
    mEnc.start_encoding(aV0);
}


void  cTest_SAC::AddCode(INT aFreq,INT aCumul,INT aTot)
{
	mEnc.encode_freq(aFreq,aCumul,aTot);

        ELISE_ASSERT(aFreq>0,"cTest_SAC::AddCode Freq");
        ELISE_ASSERT(aCumul>=0,"cTest_SAC::AddCode Cumul");
        ELISE_ASSERT(aCumul+aFreq <= aTot,"cTest_SAC::AddCode Tot");

	mVFreq.push_back(aFreq);
	mVCumul.push_back(aCumul);
	mVTot.push_back(aTot);
}

cTest_SAC::~cTest_SAC()
{
    mEnc.done_encoding();
    
    INT aV0 = mDec.start_decoding();
    ELISE_ASSERT(aV0==mV0,"cTest_SAC::~cTest_SAC");

    for (INT aK=0 ; aK<INT(mVFreq.size()) ; aK++)
    {
        INT DS_inf = mDec.decode_culfreq(mVTot[aK]);
        mDec.decode_update(mVFreq[aK],mVCumul[aK],mVTot[aK]);

        ELISE_ASSERT
        (
             (DS_inf>=mVCumul[aK])
           &&(DS_inf<(mVCumul[aK]+mVTot[aK])),
	   "cTest_SAC::~cTest_SAC Range"
        );
    }
    mDec.done_decoding();
}


void  Bench_cTest_SAC_OLD
      (
          INT v0,
          INT  freq,
          INT  s_inf,
          INT  tot,
          INT  nb
      )
{
    cTest_SAC aTS(v0);
    for (INT k=0; k< nb ; k++)
         aTS.AddCode(freq+k/2,s_inf+k,tot+2*k);
}

void  Bench_cTest_SAC_NEW(INT  nb)
{
    cTest_SAC aTS(round_ni(200*(NRrandom3())));

    for (INT k=0; k< nb ; k++)
    {
        INT aFreq = 1 + round_ni(NRrandom3()*8);
        INT aCumul = 1 + round_ni(NRrandom3()*40);
	INT aTot = round_ni((aCumul + aFreq)*(1+NRrandom3()*4));

         aTS.AddCode(aFreq,aCumul,aTot);
    }
}

void Bench_RANGE_SimpleArithmCoDec(INT aNbVal,INT aNbTest)
{
    std::vector<REAL>  mFreqs;

    for (INT aK=0 ; aK<aNbVal ; aK++)
    {
        if ((aK%4) == 0)
           mFreqs.push_back(1+NRrandom3()*20);
	else
           mFreqs.push_back(1e-3+NRrandom3());
    }

    U_INT1  aV0 = 3;
    FFOB aFlx;
    MSBF_Flux_OutVarLI       MSO(&aFlx,false);

    INT aNbBits = round_up(log(double(aNbVal))/log(2.0) + NRrandom3() * 6);
    cMS_SimpleArithmEncoder aMSAE(mFreqs,aNbBits,&MSO,aV0);
    std::vector<U_INT1> mVerifs;


    {
       const std::vector<INT> & aVCums = aMSAE.Cumuls();
       const std::vector<INT> & aVFreqs = aMSAE.Freqs();
       INT   aTot = aMSAE.Tot();
       cTest_SAC *  aTS = new cTest_SAC (aV0);


        for (INT aK=0 ; aK<aNbTest ; aK++)
        {
            INT aV = ElMin(aNbVal-1,round_ni(NRrandom3()*aNbVal));
	    mVerifs.push_back(aV);
	    aMSAE.PushCode(aV);
	    aTS->AddCode(aVFreqs[aV],aVCums[aV],aTot);
        }

	delete aTS;

    }
    aMSAE.Done();


    MSBitFirst_Flux_Of_VarLI MSI(&aFlx,false);
    cMS_SimpleArithmDecoder aMSAD(aMSAE.Cumuls(),&MSI);

    char aV1 = aMSAD.V0();
    BENCH_ASSERT(aV0==aV1);
    for (INT aK=0 ; aK<aNbTest ; aK++)
    {
        INT aV1 = mVerifs[aK];
	INT aV2 = aMSAD.Dec();
	ELISE_ASSERT(aV1==aV2,"Bench_RANGE_SimpleArithmCoDec");
    }
    aMSAD.Done();


}

void Bench_RANGE_SimpleArithmCoDec()
{
     Bench_RANGE_SimpleArithmCoDec(5,3);
     Bench_RANGE_SimpleArithmCoDec(15,2000);

     for (INT aK= 0 ; aK< 100000 ; aK++)
     {
          Bench_RANGE_SimpleArithmCoDec(round_ni(2+NRrandom3()*254),10+aK);
     }
}




void  Bench_cTest_SAC()
{
     Bench_RANGE_SimpleArithmCoDec();

     for (INT aK=0 ; aK<10000 ; aK++)
         Bench_cTest_SAC_NEW(10+aK);

     for (int S_INF = 6 ; S_INF < 18; S_INF ++)
          for (INT FREQ = 1; FREQ < 8 ; FREQ++)
                Bench_cTest_SAC_OLD(23,FREQ,S_INF,74,10000);
}


void  Bench_RANGE_CODE_1VAL_eleme
      (
          INT v0,
          INT  freq,
          INT  s_inf,
          INT  tot,
          INT  nb
      )
{
    FFOB  f;

    MSBitFirst_Flux_Of_VarLI MSI(&f,false);
    MSBF_Flux_OutVarLI       MSO(&f,false);


    MS_RANGE_ENCODER   Enc(&MSO);
    MS_RANGE_DECODER   Dec(&MSI);

    Enc.start_encoding(v0);
    for (INT k=0; k< nb ; k++)
         Enc.encode_freq(freq+k/2,s_inf+k,tot+2*k);
    Enc.done_encoding();

    INT c = Dec.start_decoding();
    BENCH_ASSERT(v0==c);
	{
    for (INT k=0; k< nb ; k++)
    {
        INT DS_inf = Dec.decode_culfreq(tot+2*k);
        Dec.decode_update(freq+k/2,s_inf+k,tot+2*k);
        BENCH_ASSERT
        (
             (DS_inf>=s_inf+k)
           &&(DS_inf<(s_inf+k+freq+k/2))
        );
    }
	}
}                              

void  Bench_RANGE_CODE_1VAL_eleme()
{
     for (int S_INF = 6 ; S_INF < 18; S_INF ++)
          for (INT FREQ = 1; FREQ < 8 ; FREQ++)
               Bench_RANGE_CODE_1VAL_eleme(23,FREQ,S_INF,74,10000);
}    




void  Bench_RANGE_CODE_Filo
      (
          INT  v0,
          ElFilo<INT>&  freq,
          ElFilo<INT>&  s_inf,
          ElFilo<INT>&  tot
      )
{
    FFOB  Stack;

    MSBitFirst_Flux_Of_VarLI MSI(&Stack,false);
    MSBF_Flux_OutVarLI       MSO(&Stack,false);


    MS_RANGE_ENCODER   Enc(&MSO);
    MS_RANGE_DECODER   Dec(&MSI);
    REAL entropy = 0.0;

    Enc.start_encoding(v0);
    for (INT k=0; k< freq.nb() ; k++)
    {
         REAL f = (REAL) freq[k] / tot[k];
         Enc.encode_freq(freq[k],s_inf[k],tot[k]);
         entropy += -El_logDeux(f);
    }
    Enc.done_encoding();
    // cout << "Entropie " << (entropy/( Stack.nb()*8)) << endl;

    INT c = Dec.start_decoding();
    BENCH_ASSERT(v0==c);
	{
    for (INT k=0; k< freq.nb() ; k++)
    {
        INT DS_inf = Dec.decode_culfreq(tot[k]);
        Dec.decode_update(freq[k],s_inf[k],tot[k]);
        BENCH_ASSERT
        (
             (DS_inf>=s_inf[k])
           &&(DS_inf<(s_inf[k]+freq[k]))
        );
    }
	}
}                                



void  Bench_RANGE_CODE_Filo()
{
     ElFilo<INT> freq;
     ElFilo<INT> s_inf;
     ElFilo<INT> tot;

     for (INT f=0 ; f< 2; f++)
     {
          for (INT k=0 ; k< 100000 ; k++)
          {
              s_inf.pushlast(1+k/10);
              freq.pushlast(s_inf.top()*30);
              tot.pushlast((1+(k%3))*s_inf.top()+freq.top());
          }
          Bench_RANGE_CODE_Filo(33,freq,s_inf,tot);
		  {
          for (INT k=0 ; k< 10000 ; k++)
          {
              freq.pushlast(1+k/10);
              s_inf.pushlast(freq.top()*30);
              tot.pushlast((1+(k%3))*s_inf.top()+freq.top());
          }
		  }
          Bench_RANGE_CODE_Filo(33,freq,s_inf,tot);
     }
}







void  Bench_RANGE_CODE()
{
      Bench_RANGE_CODE_Filo();
      Bench_RANGE_CODE_1VAL_eleme();
   cout << "OK RANGE CODE \n";
}                             



void bench_bits_flow()
{
     Bench_cTest_SAC();
     Bench_RANGE_CODE();
     bench_msb_varLI();
     bench_lsb_varLI();
     bench_msb_out_varLI();
     bench_lsb_out_varLI();


     cout << "OK bits flows \n";
}
