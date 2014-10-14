#ifndef __TIMES__
#define __TIMES__

#define __DEBUG_TIMES

class Times
{
public:
	virtual void reset() = 0;
	virtual void start() = 0;
	virtual double stop( const char *i_name ) = 0;
	virtual double get( const std::string &i_name ) const = 0;
	virtual ~Times();
};

class NoTimes : public Times
{
public:
	#ifdef __DEBUG_TIMES
		bool mIsStarted;

		inline NoTimes();
		inline ~NoTimes();
	#endif

	inline void reset();
	inline void start();
	inline double stop( const char *i_name );
	inline double get( const std::string &i_name ) const;
};

class MapTimes : public Times
{
public:
	ElTimer            mTimer;
	map<string,double> mRecords;

	inline MapTimes();
	inline void reset(); // reset all times (clear the map)
	inline void start();
	double stop( const char *i_name );  // stop time and add it to the record of name i_name
	double get( const std::string &i_name ) const;
	inline const std::map<std::string,double> & getAllTimes() const;
	double totalTime() const;
	void printTimes( const std::string &i_prefix=std::string(), std::ostream &io_ostream=std::cout ) const;
};

#include "Times.inline.h"

#endif // __TIMES__
