#pragma once

#include "deal.II/base/exceptions.h"

#include <boost/core/null_deleter.hpp>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/sinks/sync_frontend.hpp>
#include <boost/log/sinks/text_ostream_backend.hpp>
#include <boost/log/sources/global_logger_storage.hpp>
#include <boost/log/sources/record_ostream.hpp>
#include <boost/log/sources/severity_logger.hpp>
#include <boost/log/support/date_time.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>

#include <iostream>
#include <regex>
#include <sstream>
#include <string>

namespace Ddhdg
{
  namespace Logging
  {
    DeclExceptionMsg(InvalidSeverityLevel,
                     "Invalid logging severity level specified");

    enum severity_level
    {
      trace   = 0,
      debug   = 1,
      info    = 2,
      warning = 3,
      error   = 4,
      fatal   = 5
    };

    inline std::string
    severity_level_2_str(const severity_level level)
    {
      switch (level)
        {
          case (severity_level::trace):
            return "trace";
          case (severity_level::debug):
            return "debug";
          case (severity_level::info):
            return "info";
          case (severity_level::warning):
            return "warning";
          case (severity_level::error):
            return "error";
          case (severity_level::fatal):
            return "fatal";
          default:
            Assert(false, InvalidSeverityLevel());
            break;
        }
      return "unknown";
    }

    BOOST_LOG_ATTRIBUTE_KEYWORD(severity, "Severity", severity_level)

    inline void
    log_formatter(const boost::log::record_view & view,
                  boost::log::formatting_ostream &os)
    {
      const auto level_ref =
        view.attribute_values()["Severity"].extract<severity_level>();
      std::string level;
      if (!level_ref.empty())
        level = severity_level_2_str(*level_ref);
      else
        level = "UNKNOWN";

      std::string timestamp_str = "NO_DATE";

      const auto timestamp = view.attribute_values()["TimeStamp"]
                               .extract<boost::posix_time::ptime>();
      if (!timestamp.empty())
        {
          boost::posix_time::time_facet *facet =
            new boost::posix_time::time_facet();
          facet->format("%Y-%m-%d %H:%M:%S");
          std::stringstream stream;
          stream.imbue(std::locale(std::locale::classic(), facet));
          stream << *timestamp;
          timestamp_str = stream.str();
        }

      os << "[" << timestamp_str << "] [" << level
         << "]: " << view.attribute_values()["Message"].extract<std::string>();
    }

    typedef boost::log::sources::severity_logger_mt<severity_level> l_type;

    BOOST_LOG_INLINE_GLOBAL_LOGGER_INIT(LOGGER, l_type)
    {
      boost::log::sources::severity_logger_mt<severity_level> lg;

      typedef boost::log::sinks::synchronous_sink<
        boost::log::sinks::text_ostream_backend>
                                   text_sink;
      boost::shared_ptr<text_sink> sink = boost::make_shared<text_sink>();

      boost::shared_ptr<std::ostream> out_stream{&std::clog,
                                                 boost::null_deleter{}};
      sink->locked_backend()->add_stream(out_stream);

      sink->set_formatter(&log_formatter);
      sink->set_filter(severity >= severity_level::info);

      boost::log::add_common_attributes();

      boost::log::core::get()->add_sink(sink);
      return lg;
    }

    inline unsigned int
    n_of_occurrences(const std::string &pattern, const std::string &s)
    {
      unsigned int           occurrences = 0;
      std::string::size_type pos         = 0;
      while ((pos = s.find(pattern, pos)) != std::string::npos)
        {
          ++occurrences;
          pos += pattern.length();
        }
      return occurrences;
    }



    template <severity_level level>
    inline void
    log(const std::string &log_message)
    {
      if constexpr (level == severity_level::trace)
        {
          BOOST_LOG_SEV(LOGGER::get(), severity_level::trace) << log_message;
        }
      else
        BOOST_LOG_SEV(LOGGER::get(), level) << log_message;
    }

    inline void
    log(const std::string &log_message, const severity_level level)
    {
      switch (level)
        {
          case (severity_level::trace):
            log<severity_level::trace>(log_message);
            break;
          case (severity_level::debug):
            log<severity_level::debug>(log_message);
            break;
          case (severity_level::info):
            log<severity_level::info>(log_message);
            break;
          case (severity_level::warning):
            log<severity_level::warning>(log_message);
            break;
          case (severity_level::error):
            log<severity_level::error>(log_message);
            break;
          case (severity_level::fatal):
            log<severity_level::fatal>(log_message);
            break;
          default:
            Assert(false, InvalidSeverityLevel());
            break;
        }
    }

    template <severity_level level>
    inline void
    log(const std::string &log_message, const unsigned int n)
    {
      Assert(n_of_occurrences("%s", log_message) == 1,
             dealii::ExcMessage(
               "Wrong number of \"%s\" inside the log message"));
      const std::string str_n = std::to_string(n);
      const std::string new_log_message =
        std::regex_replace(log_message, std::regex("%s"), str_n);
      log<level>(new_log_message);
    }

    inline void
    log(const std::string &  log_message,
        const unsigned int   n,
        const severity_level level)
    {
      switch (level)
        {
          case (severity_level::trace):
            log<severity_level::trace>(log_message, n);
            break;
          case (severity_level::debug):
            log<severity_level::debug>(log_message, n);
            break;
          case (severity_level::info):
            log<severity_level::info>(log_message, n);
            break;
          case (severity_level::warning):
            log<severity_level::warning>(log_message, n);
            break;
          case (severity_level::error):
            log<severity_level::error>(log_message, n);
            break;
          case (severity_level::fatal):
            log<severity_level::fatal>(log_message, n);
            break;
          default:
            Assert(false, InvalidSeverityLevel());
            break;
        }
    }

    template <severity_level level>
    inline void
    log(const std::string &log_message, int n)
    {
      Assert(n_of_occurrences("%s", log_message) == 1,
             dealii::ExcMessage(
               "Wrong number of \"%s\" inside the log message"));
      const std::string str_n = std::to_string(n);
      const std::string new_log_message =
        std::regex_replace(log_message, std::regex("%s"), str_n);
      log<level>(new_log_message);
    }

    inline void
    log(const std::string &log_message, const int n, const severity_level level)
    {
      switch (level)
        {
          case (severity_level::trace):
            log<severity_level::trace>(log_message, n);
            break;
          case (severity_level::debug):
            log<severity_level::debug>(log_message, n);
            break;
          case (severity_level::info):
            log<severity_level::info>(log_message, n);
            break;
          case (severity_level::warning):
            log<severity_level::warning>(log_message, n);
            break;
          case (severity_level::error):
            log<severity_level::error>(log_message, n);
            break;
          case (severity_level::fatal):
            log<severity_level::fatal>(log_message, n);
            break;
          default:
            Assert(false, InvalidSeverityLevel());
            break;
        }
    }

    template <severity_level level>
    inline void
    log(const std::string &log_message, double n)
    {
      Assert(n_of_occurrences("%s", log_message) == 1,
             dealii::ExcMessage(
               "Wrong number of \"%s\" inside the log message"));
      std::ostringstream stream_n;
      stream_n << n;
      const std::string str_n = stream_n.str();
      const std::string new_log_message =
        std::regex_replace(log_message, std::regex("%s"), str_n);
      log<level>(new_log_message);
    }

    inline void
    log(const std::string &  log_message,
        const double         n,
        const severity_level level)
    {
      switch (level)
        {
          case (severity_level::trace):
            log<severity_level::trace>(log_message, n);
            break;
          case (severity_level::debug):
            log<severity_level::debug>(log_message, n);
            break;
          case (severity_level::info):
            log<severity_level::info>(log_message, n);
            break;
          case (severity_level::warning):
            log<severity_level::warning>(log_message, n);
            break;
          case (severity_level::error):
            log<severity_level::error>(log_message, n);
            break;
          case (severity_level::fatal):
            log<severity_level::fatal>(log_message, n);
            break;
          default:
            Assert(false, InvalidSeverityLevel());
            break;
        }
    }
  } // namespace Logging
} // namespace Ddhdg
