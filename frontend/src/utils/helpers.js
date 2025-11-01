export const buildFormData = (file) => {
  const formData = new FormData();
  formData.append("file", file);
  return formData;
};

export const toast = (opts) => {
  if (!opts) return;
  if (opts.status === "error") console.error(opts.title, opts.description);
  else console.log(opts.title, opts.description);
};