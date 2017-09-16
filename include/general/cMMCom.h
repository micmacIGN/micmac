#ifndef __C_MM_COM__
#define __C_MM_COM__

typedef int (*tCommande) (int,char**);

class cArgLogCom
{
    public :

        cArgLogCom(int aNumArg,const std::string &aDirSup = "") :
            mNumArgDir (aNumArg),
            mDirSup    (aDirSup)
        {
        }

        int mNumArgDir ;
        std::string  mDirSup;

        static const cArgLogCom NoLog;
};

// CMMCom is a descriptor of a MicMac Command
class cMMCom
{
public :
	cMMCom(const std::string &aName, tCommande aCommand, const std::string &aComment, const cArgLogCom &aLog = cArgLogCom::NoLog):
		mName(aName),
		mLowName(StrToLower(aName)),
		mCommand(aCommand),
		mComment(aComment),
		mLog(aLog)
	{
	}

	std::string  mName;
	std::string  mLowName;
	tCommande    mCommand;
	std::string  mComment;
	std::string  mLib;
	cArgLogCom  mLog;
};

#endif
