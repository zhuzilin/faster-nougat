import os
from argparse import ArgumentParser
from tqdm import tqdm


from faster_nougat.utils import get_model_and_processor, extract_pdf_pages_as_images
from faster_nougat import generate

parser = ArgumentParser()
parser.add_argument(
    "--arxiv_url",
    type=str,
    required=True,
    help="The pdf url of arxiv paper, e.g. https://arxiv.org/pdf/1706.03762",
)
parser.add_argument(
    "--cache_dir",
    type=str,
    default="cache/",
    help="The dir to save the downloaded paper and the parse result",
)
parser.add_argument("--nougat_hf_model_name", type=str, default="facebook/nougat-small")
parser.add_argument(
    "--pages",
    type=int,
    nargs="+",
    default=None,
    help="The pages to read, default will process all pages, starts from 1",
)
parser.add_argument(
    "--ignore_pages",
    type=int,
    nargs="+",
    default=None,
    help="The pages to ignore, default will process all pages, starts from 1",
)
parser.add_argument(
    "--resolution", type=int, default=200, help="the resolution of pdf image"
)
# llm configs
parser.add_argument("--llm_model_name", type=str, default="deepseek-chat")
parser.add_argument("--llm_base_url", type=str, default="https://api.deepseek.com/")
parser.add_argument("--llm_key", type=str, default=None)
parser.add_argument(
    "--question", type=str, default="could you tell me what does this paper contribute?"
)
parser.add_argument("-f", action="store_true", help="Force to regenerate")


if __name__ == "__main__":
    args = parser.parse_args()

    os.makedirs(args.cache_dir, exist_ok=True)

    pdf_path = os.path.join(args.cache_dir, args.arxiv_url.split("/")[-1])
    if not pdf_path.endswith(".pdf"):
        pdf_path += ".pdf"

    # download pdf
    if not os.path.exists(pdf_path):
        os.system(f"wget {args.arxiv_url} -O {pdf_path}")

    if args.pages is not None:
        images = None
        args.pages = [page_idx - 1 for page_idx in args.pages]
    else:
        images = extract_pdf_pages_as_images(pdf_path)
        args.pages = list(range(len(images)))

    if args.ignore_pages is not None:
        args.pages = [
            page_idx for page_idx in args.pages if page_idx not in args.ignore_pages
        ]

    model, processor = None, None
    warning_for_exist_files = True
    for page_idx in tqdm(args.pages):

        page_path = os.path.join(args.cache_dir, f"page_{page_idx + 1}.mmd")
        if os.path.exists(page_path) and not args.f:
            if warning_for_exist_files:
                print(
                    f"some output file exists, will skip. please pass -f to force overwrite"
                )
                warning_for_exist_files = False
            continue

        if images is None:
            images = extract_pdf_pages_as_images(pdf_path, args.resolution)
        image = images[page_idx]
        if model is None:
            model, processor = get_model_and_processor(args.nougat_hf_model_name)

        pixel_values = processor(image, return_tensors="pt").pixel_values
        outputs = generate(model, pixel_values, max_new_tokens=4096, disable_tqdm=True)
        sequence = processor.batch_decode([outputs], skip_special_tokens=True)[0]
        sequence = processor.post_process_generation(sequence, fix_markdown=False)
        with open(page_path, "w") as f:
            f.write(sequence)

    if args.llm_key is None:
        print("llm_key not set, exit.")
        exit()

    from openai import OpenAI

    client = OpenAI(api_key=args.llm_key, base_url=args.llm_base_url)

    prompt = ""
    for page_idx in args.pages:
        prompt += "\n" + "-" * 10 + f" PAGE {page_idx + 1} " + "-" * 10 + "\n"
        page_path = os.path.join(args.cache_dir, f"page_{page_idx + 1}.mmd")
        with open(page_path) as f:
            prompt += f.read()
    prompt += "\n" + "-" * 20 + "\n"
    prompt += args.question

    print(args.question)
    print("-" * 20)

    response = client.chat.completions.create(
        model=args.llm_model_name,
        messages=[
            {
                "role": "system",
                "content": "You will be given some pages of an paper, with ---- PAGE x ---- separated. "
                "Please use the content in the paper and your own knowledge to provide helpful answer the question.",
            },
            {"role": "user", "content": prompt},
        ],
        stream=True,
    )

    answer = ""
    for chunk in response:
        answer_section = chunk.choices[0].delta.content
        answer += answer_section
        print(answer_section, end="", flush=True)
