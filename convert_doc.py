import json
class ConvertDoc:

    
    def convertToUtf8( input_file, output_file):
        with open(input_file, 'r', encoding='cp1252', errors='ignore') as f_in:
            content = f_in.read()

        with open(output_file, 'w', encoding='utf-8') as f_out:
            f_out.write(content) 
        print(f"تم تحويل {input_file} إلى {output_file}")




    def convert_jsonl_to_json(input_file, output_file):
        data = []
        with open(input_file, "r", encoding="utf-8") as fin:
            for line in fin:
                data.append(json.loads(line.strip()))

        with open(output_file, "w", encoding="utf-8") as fout:
            json.dump(data, fout, ensure_ascii=False, indent=2)

        print(f"✅ Converted {input_file} to {output_file} (as JSON array)")
    

    def json_to_tsv(json_file_path, tsv_file_path):
        with open(json_file_path, 'r', encoding='utf-8') as infile:
            data = json.load(infile)

        with open(tsv_file_path, 'w', encoding='utf-8') as outfile:
            if isinstance(data, dict):
                # إذا كان json عبارة عن قاموس {id: text}
                for doc_id, text in data.items():
                    outfile.write(f"{doc_id}\t{text}\n")
            elif isinstance(data, list):
                # إذا كان json عبارة عن قائمة [{_id:..., text:...}, ...]
                for item in data:
                    doc_id = item.get("_id") or item.get("id")
                    text = item.get("text", "")
                    if doc_id is not None:
                        outfile.write(f"{doc_id}\t{text}\n")
                    else:
                        raise ValueError("Unsupported JSON structure")
            else:
                raise ValueError("Unsupported JSON structure")

        print(f"✅ تم تحويل {json_file_path} إلى {tsv_file_path}")




                