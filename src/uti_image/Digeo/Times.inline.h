// this file should be included by Times.h solely

//----------------------------------------------------------------------
// methods of class Times
//----------------------------------------------------------------------

inline Times::~Times(){}


//----------------------------------------------------------------------
// methods of class NoTimes
//----------------------------------------------------------------------

#ifdef __DEBUG_TIMES
	NoTimes::NoTimes():mIsStarted(false){}

	NoTimes::~NoTimes()
	{
		if ( mIsStarted )
			__elise_debug_error( "NoTimes::~NoTimes: destructing started NoTimes" );
	}

	void NoTimes::start()
	{
		if ( mIsStarted )
			__elise_debug_error( "NoTimes::start: starting an already started NoTimes" );
		mIsStarted = true;
	}

	double NoTimes::stop( const char *i_name )
	{
		if ( !mIsStarted )
			__elise_debug_error( "NoTimes::stop: stopping a NoTimes that is not started" );
		mIsStarted = false;
		return 0.;
	}
#else
	double NoTimes::stop( const char *i_name ){ return 0.; }
	void NoTimes::start(){}
#endif

void NoTimes::reset(){}

double NoTimes::get( const std::string &i_name ) const { return 0.; }


//----------------------------------------------------------------------
// methods of class NoTimes
//----------------------------------------------------------------------

MapTimes::MapTimes(){}

void MapTimes::reset() { mRecords.clear(); }

void MapTimes::start()
{
	mTimer.reinit();
}

const std::map<std::string,double> & MapTimes::getAllTimes() const { return mRecords; }
