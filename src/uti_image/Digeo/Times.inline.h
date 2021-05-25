// this file should be included by Times.h solely

//----------------------------------------------------------------------
// methods of class Times
//----------------------------------------------------------------------

inline Times::~Times(){}


//----------------------------------------------------------------------
// methods of class NoTimes
//----------------------------------------------------------------------

inline double NoTimes::stop( const char *i_name ){ return 0.; }

inline void NoTimes::start(){}

inline void NoTimes::clear(){}


//----------------------------------------------------------------------
// methods of class MapTimes::Record
//----------------------------------------------------------------------

inline MapTimes::Record::Record(){}


//----------------------------------------------------------------------
// methods of class MapTimes
//----------------------------------------------------------------------

inline void MapTimes::clear() { mRecords.clear(); }

inline double MapTimes::totalTime() const { return mRecords.begin()->totalTime(); }

#ifndef ELISE_unix
	inline Timer::Timer():mRecordedTime(getTime()){}

	inline void Timer::reinit(){ mRecordedTime = getTime(); }

	inline double Timer::uval(){ return getTime()-mRecordedTime; }
#endif
