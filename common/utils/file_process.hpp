//
// Created by Li Zhaoliang on 2023/8/2.
//

#include <algorithm>
#include <istream>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace alg_utils
{
    std::vector<std::string> split(char separator, const std::string &string, bool ignore_empty)
    {
        std::vector<std::string> pieces;
        std::stringstream ss(string);
        std::string item;
        while (getline(ss, item, separator))
        {
            if (!ignore_empty || !item.empty())
            {
                pieces.push_back(std::move(item));
            }
        }
        return pieces;
    }

    std::string trim(const std::string &str)
    {
        size_t left = str.find_first_not_of(' ');
        if (left == std::string::npos)
        {
            return str;
        }
        size_t right = str.find_last_not_of(' ');
        return str.substr(left, (right - left + 1));
    }

    inline bool StartsWith(
            const std::string &str,
            const std::string &prefix)
    {
        return str.length() >= prefix.length() &&
               std::mismatch(prefix.begin(), prefix.end(), str.begin()).first ==
               prefix.end();
    }

    inline bool EndsWith(
            const std::string &full,
            const std::string &ending)
    {
        if (full.length() >= ending.length())
        {
            return (
                    0 ==
                    full.compare(full.length() - ending.length(), ending.length(), ending));
        } else
        {
            return false;
        }
    }

    void get_all_line_from_txt(const std::string &txt_path,
                               std::vector<std::string> &all_lines)
    {
        all_lines.clear();
        std::ifstream infile(txt_path, std::ios::in);
        std::string line;
        while (getline(infile, line))
        {
            all_lines.emplace_back(line);
        }
        infile.close();
        return;
    }
}