for file in (git ls-files 'src/*.py'); do 
    echo "========================================";
    echo "FILE: $file";
    echo "========================================";
    cat "$file";
    echo -e "\n";
done > projet_sailing_dump.txt